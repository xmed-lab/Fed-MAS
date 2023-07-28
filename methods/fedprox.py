'''
Code credit to https://github.com/QinbinLi/MOON
for thier implementation of FedProx.
'''

import torch
import logging

from methods.utils import adjust_learning_rate
import copy
from torch.multiprocessing import current_process
from methods.fedavg import Client as fedavg_Client
from methods.fedavg import Server as fedavg_Server




class Client(fedavg_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)

    def train(self):
        # train the local model
        self.model.to(self.device)
        self.optimizer.load_state_dict(self.optimizer_client_states[self.client_index])

        global_weight_collector = copy.deepcopy(list(self.model.parameters()))
        self.model.train()
        epoch_loss = []
        if self.args.adjust_learning_rate:
            adjust_learning_rate(self.optimizer, self.round, self.args.lr,self.args.comm_round)

        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                _,log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                ############
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += (
                                (self.args.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss = loss + fed_prox_reg
                ########
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}  Thread {}  Map {}'.format(self.client_index,
                                                                                                    epoch, sum(
                            epoch_loss) / len(epoch_loss), current_process()._identity[0], self.client_map[self.round]))
        self.optimizer_client_states[self.client_index] = copy.deepcopy(self.optimizer.state_dict())
        weights = self.model.cpu().state_dict()
        return weights


class Server(fedavg_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
