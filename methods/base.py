import copy
import os
import sys

import pandas as pd
import torch
import logging

from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score

from methods.utils import adjust_learning_rate, ComboLoader

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

from torch.multiprocessing import current_process
import numpy as np


class Base_Client():
    def __init__(self, client_dict, args):
        self.train_data = client_dict['train_data']
        self.criterion_dict = client_dict['criterion']
        self.val_data = client_dict['val_data']
        self.test_data = client_dict['test_data']
        self.device = 'cuda:{}'.format(client_dict['device'])
        self.model_type = client_dict['model_type']
        self.num_classes = client_dict['num_classes']
        self.client_best_weights = {}
        self.client_best_fusion = {}
        self.args = args
        self.round = 0
        self.test_file_path = client_dict['test_out_path']
        self.client_map = client_dict['client_map']
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.best_acc = {i: 0 for i in range(args.client_number)}
        self.best_acc_crt = {i: 0 for i in range(args.client_number)}
        self.client_index = None

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.client_index = client_idx

            # if you want to update clients in the fed-setting or each client is independent (local learning)
            if self.args.update_clients == 1:
                self.load_client_state_dict(received_info['global'])

            self.train_dataloader = self.train_data[client_idx]
            self.training_labels = self.train_dataloader.dataset.target

            self.test_dataloader = self.test_data[client_idx]
            self.val_dataloader = self.val_data[client_idx]

            self.criterion = self.criterion_dict[client_idx]
            self.criterion.to(self.device)
            self.class_priors = self.get_class_priors()

            # sampling of clients according to specific ratio if enabeled
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()

            num_samples = len(self.train_dataloader) * self.args.batch_size

            weights = self.train()

            acc = -1
            if self.round % self.args.validate_client_every == 0:
                # if saving client weights/evaluate it and save the best weights according to the client val set
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
            logger.info(f"** Client: {self.client_index}, Acc = {round(acc, 4)} "
                        f"***")

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

        # for different clients on the same thread
        self.optimizer.load_state_dict(self.optimizer_client_states[self.client_index])

        self.model.train()

        epoch_loss = []


        if self.args.adjust_learning_rate:
            adjust_learning_rate(self.optimizer, self.round, self.args.lr, self.args.comm_round,
                                 multi_step=self.args.multi_step)

        if self.args.loss_fn_name == 'LDAM':
            self.criterion.set_weight(self.round)

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                _, log_probs = self.model(images)

                loss = self.criterion(log_probs, labels)

                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logger.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}, LR:{} Thread {}  Map {}'.format(
                        self.client_index,
                        epoch, sum(
                            epoch_loss) / len(epoch_loss), self.optimizer.param_groups[0]['lr'],
                        current_process()._identity[0], self.client_map[self.round]))

        self.optimizer_client_states[self.client_index] = copy.deepcopy(self.optimizer.state_dict())
        weights = self.model.cpu().state_dict()
        return weights

    def get_class_priors(self):
        class_count_per_client = [0] * (self.num_classes)
        class_idx, class_count_per_client_tr = np.unique(self.train_dataloader.dataset.target,
                                                         return_counts=True)
        for idx, j in enumerate(class_idx):
            class_count_per_client[j] = class_count_per_client_tr[idx]

        class_count_per_client /= np.sum(class_count_per_client)
        return dict(zip(range(len(class_count_per_client)), class_count_per_client))

    def test(self):
        if self.args.validate_client and self.round == self.args.comm_round - 1:
            save_client = True
            logger.info(f'Loading Best Model for Client {self.client_index}')
            self.model.load_state_dict(self.client_best_weights[self.client_index])
            test_data = self.val_dataloader
        else:
            save_client = False
            test_data = self.test_dataloader

        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_sample_number = 0.0
        pred_list = []
        target_list = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                features, pred = self.model(x)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_sample_number += target.size(0)

                pred_list.extend(predicted.cpu().detach().tolist())
                target_list.extend(target.cpu().detach().tolist())

        if save_client:
            d = {'pred': pred_list, 'target': target_list}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(self.test_file_path, f'{self.client_index}.csv'))

        acc = balanced_accuracy_score(target_list, pred_list)
        return acc


class Base_Server():
    def __init__(self, server_dict, args):
        self.train_data = server_dict['train_data']
        self.test_data = server_dict['test_data']
        self.val_data = server_dict['val_data']

        self.device = 'cuda:{}'.format(torch.cuda.device_count() - 1)
        self.model_type = server_dict['model_type']
        self.num_classes = server_dict['num_classes']
        self.acc = 0.0
        self.round = 0
        self.args = args
        self.save_path = server_dict['save_path']
        self.test_file_path = server_dict['test_out_path']

    def run(self, received_info):
        server_outputs = self.operations(received_info)

        validate = False
        if self.round == self.args.comm_round - 1:
            self.model.load_state_dict(self.best_weights)
            logger.info("************* Loading best weights For Test Set **************")
            validate = True

        acc = self.test(validate)

        logger.info("************* Server Acc = {:.2f} **************".format(acc))

        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.acc = acc
        server_outputs = [{'global': g['weights'],
                           'server_acc': acc,
                           } for g in server_outputs]
        return server_outputs

    def start(self):
        return [{'global': self.model.cpu().state_dict(), 'server_acc': 0} for x in
                range(self.args.thread_number)]

    def log_info(self, client_info, acc):
        client_acc = sum([c['acc'] for c in client_info]) / len(client_info)
        out_str = 'Test/AccTop1: {}, Client_Train/AccTop1: {}, round: {}\n'.format(acc, client_acc, self.round)
        with open('{}/out.log'.format(self.save_path), 'a+') as out_file:
            out_file.write(out_str)

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        # logger.info(len(client_sd))
        cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]

        ssd = self.model.state_dict()
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])
        self.model.load_state_dict(ssd)

        #
        return [{'weights': self.model.cpu().state_dict()
                 } for
                x in range(self.args.thread_number)]

    def test(self, validate=False):
        self.model.to(self.device)
        self.model.eval()
        test_correct = 0.0
        test_sample_number = 0.0

        pred_list = []
        target_list = []
        if validate:
            test_data = self.val_data
        else:
            test_data = self.test_data

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                features, pred = self.model(x)
                # loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
                pred_list.extend(predicted.cpu().detach().tolist())
                target_list.extend(target.cpu().detach().tolist())

        if validate:
            d = {'pred': pred_list, 'target': target_list}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(self.test_file_path, 'server.csv'))
        acc = balanced_accuracy_score(target_list, pred_list)
        return acc
