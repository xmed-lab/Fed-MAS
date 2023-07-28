'''
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
'''
import copy
import os
import sys
from pathlib import Path

import torch
import logging


from methods.utils import ComboLoader, get_pre_trained_model_dict, ClassifierLDAM

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_model = self.model_type(self.num_classes, KD=True, projection=True)
        self.global_model = self.model_type(self.num_classes, KD=True, projection=True)

        # list(self.encoder_model.children())[-1].in_features
        if self.args.pre_trained_tv:
            new_state_dict = get_pre_trained_model_dict(args, self.model.state_dict())
            # load pre_trained model succesfully
            self.model.load_state_dict(new_state_dict)

        self.feat_dim = list(self.model.children())[-1].in_features

        if self.args.loss_fn_name == 'LDAM':
            self.model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)

        if self.args.adamw:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif args.adam_w:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, wd=self.args.wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,
                                             weight_decay=self.args.wd, nesterov=True)

        self.optimizer_client_states = {i: self.optimizer.state_dict() for i in client_dict['client_map'][0]}


        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temp = 0.5


    def run(self, received_info):
        client_results = []
        self.global_model.load_state_dict(received_info['global'])
        for client_idx in self.client_map[self.round]:
            self.client_index = client_idx

            self.prev_model.load_state_dict(received_info['prev'][client_idx])
            self.load_client_state_dict(received_info['global'])


            # getting all the client information from the queue dict
            self.train_dataloader = self.train_data[client_idx]
            self.training_labels = self.train_dataloader.dataset.target

            self.test_dataloader = self.test_data[client_idx]
            self.val_dataloader = self.val_data[client_idx]

            self.criterion = self.criterion_dict[client_idx]
            self.criterion_ce = torch.nn.CrossEntropyLoss().to(self.device)

            self.criterion.to(self.device)
            self.class_priors = self.get_class_priors()

            # sampling of clients according to specific ratio if enabeled
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()

            num_samples = len(self.train_dataloader) * self.args.batch_size

            weights = self.train()

            acc = -1
            if self.round % self.args.validate_client_every == 0:

                if self.args.validate_client or self.args.save_client:
                    if self.round == 0:
                        self.client_best_weights[self.client_index] = copy.deepcopy(weights)
                    acc = self.test()
                    if acc >= self.best_acc[self.client_index]:
                        self.best_acc[self.client_index] = acc
                        self.client_best_weights[self.client_index] = copy.deepcopy(weights)

            # save client best weights in the last round
            if self.args.save_client and self.round == self.args.comm_round - 1:
                torch.save(self.client_best_weights[self.client_index],
                           f'logs/{self.parition_path}/teacher_{self.client_index}.pt')

            # logging client info
            logger.info(f"** Client: {self.client_index}, Acc = {round(acc, 4)}***")

            # appending client info to the queue
            client_results.append(
                {'weights': copy.deepcopy(weights),
                 'num_samples': num_samples,
                 'acc': acc,
                 'client_index': self.client_index})
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.optimizer.load_state_dict(self.optimizer_client_states[self.client_index])

        self.global_model.to(self.device)
        self.prev_model.to(self.device)
        self.model.train()
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logging.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                #####
                pro1, out = self.model(x)
                pro2, _ = self.global_model(x)

                posi = self.cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                pro3, _ = self.prev_model(x)
                nega = self.cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.temp
                labels = torch.zeros(x.size(0)).to(self.device).long()

                loss2 = self.args.mu * self.criterion_ce(logits, labels)

                loss1 = self.criterion(out, target)
                loss = loss1 + loss2
                #####
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logger.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        self.optimizer_client_states[self.client_index] = copy.deepcopy(self.optimizer.state_dict())
        weights = self.model.cpu().state_dict()
        return weights


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True, projection=True)
        self.prev_models = {x: self.model.cpu().state_dict() for x in range(self.args.client_number)}

        self.feat_dim = list(self.model.children())[-1].in_features

        if self.args.loss_fn_name == 'LDAM':
            self.model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)

        if self.args.pre_trained_tv:
            new_state_dict = get_pre_trained_model_dict(args, self.model.state_dict())
            # load pre_trained model succesfully
            self.model.load_state_dict(new_state_dict)

        self.best_weights = copy.deepcopy(self.model.state_dict())

    def run(self, received_info):
        server_outputs = self.operations(received_info)

        validate = False
        if self.round == self.args.comm_round - 1:
            self.model.load_state_dict(self.best_weights)
            logger.info("************* Loading best weights For Test Set of EyePacs **************")
            validate = True

        acc = self.test(validate)

        logger.info("************* Server Acc = {:.2f} **************".format(acc))

        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:

            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.acc = acc

        for x in received_info:
            self.prev_models[x['client_index']] = x['weights']
        server_outputs = [{'global': g['weights'],
                           'prev': self.prev_models,
                           'server_acc': acc,
                           } for g in server_outputs]
        return server_outputs

        #
    def start(self):
        return [{'global': self.model.cpu().state_dict(), 'prev': self.prev_models, 'server_acc': 0} for x in
                range(self.args.thread_number)]

