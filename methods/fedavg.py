import copy
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from methods.utils import BlSoftmaxLoss, FocalLoss, LDAMLoss, ClassifierLDAM, get_pre_trained_model_dict
from methods.base import Base_Client, Base_Server

import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.model = self.model_type(self.num_classes, KD=True).to(self.device)
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

        #wont work if u want to add sampling clients
        self.optimizer_client_states = {i: self.optimizer.state_dict() for i in client_dict['client_map'][0]}

class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        self.model = self.model_type(self.num_classes, KD=True)
        self.feat_dim = list(self.model.children())[-1].in_features
        if self.args.loss_fn_name == 'LDAM':
            self.model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)

        if self.args.pre_trained_tv:
            new_state_dict = get_pre_trained_model_dict(args, self.model.state_dict())
            # load pre_trained model succesfully
            self.model.load_state_dict(new_state_dict)


        self.best_weights = copy.deepcopy(self.model.state_dict())
