'''
Code credit to https://github.com/QinbinLi/MOON
for implementation of thier method, MOON.
'''
import copy
import os
import sys
# import clip
import pandas as pd

import torch
import logging

from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
from methods.utils import adjust_learning_rate, get_effective_number_weight, get_pre_trained_model_dict, \
    ClassifierLDAM
from methods.base import Base_Client, Base_Server
from torch.multiprocessing import current_process
from torchvision.models import resnet50, ResNet50_Weights, ResNet18_Weights, resnet18

import torch.nn.functional as F
from torch import nn
import numpy as np
from methods.ramp import LinearRampUp
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

class Client(Base_Client):
    def __init__(self, client_dict, args):
        super().__init__(client_dict, args)
        self.free_factor = args.free_u
        self.distill_factor = args.distill_u
        self.ramp_up = LinearRampUp(length=args.num_warmup_rounds)
        self.ramp_up_free = LinearRampUp(length=args.comm_round - args.start_free_rd)
        self.pre_Trained_model_type = client_dict['pre_trained']

        if client_dict['pre_trained'] == 'moco':
            free_model_output_dim = 2048
            logger.info('loading moco model')
            checkpoint = torch.load('./data/moco_v2_800ep_pretrain.pth.tar')
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q.'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            self.pre_trained_model = resnet50(weights=None, num_classes=128)
            dim_mlp = self.pre_trained_model.fc.weight.shape[1]
            self.pre_trained_model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.pre_trained_model.fc)
            self.pre_trained_model.load_state_dict(state_dict)
            self.pre_trained_model.fc_2 = self.pre_trained_model.fc
            self.pre_trained_model.fc = torch.nn.Identity()
        elif client_dict['pre_trained'] == 'clip':
            raise Exception('Download clip dependency')
            # self.pre_Trained_model_type = 'clip'
            # logger.info('loading CLIP model')
            # self.pre_trained_model, preprocess = clip.load("ViT-B/32", device=self.device)
            # self.pre_trained_transform = Compose([
            #     Resize(self.pre_trained_model.visual.input_resolution, interpolation=BICUBIC),
            #     CenterCrop(self.pre_trained_model.visual.input_resolution),
            #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            # ])
            # free_model_output_dim = 512
        elif client_dict['pre_trained'] == 'dinov2':
            self.pre_Trained_model_type = 'dinov2'
            logger.info('loading dinov2 model')
            self.pre_trained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.pre_trained_model.fc = self.pre_trained_model.head
            self.pre_trained_model.fc_2 = self.pre_trained_model.head
            free_model_output_dim = 384
        elif client_dict['pre_trained'] == 'IMAGENET1K_V1':
            self.pre_trained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, num_classes=1000)
            self.pre_trained_model.fc_2 = self.pre_trained_model.fc
            self.pre_trained_model.fc = torch.nn.Identity()
            free_model_output_dim = 2048
        elif client_dict['pre_trained'] == 'radImageNet':
            self.pre_trained_model = resnet50(weights=None, num_classes=1)
            self.pre_trained_model.load_state_dict(torch.load('./data/RadImageNet-ResNet50_notop.pth'))
            self.pre_trained_model.fc_2 = self.pre_trained_model.fc
            self.pre_trained_model.fc = torch.nn.Identity()
            free_model_output_dim = 2048
        elif client_dict['pre_trained'] == '18_img':
            self.pre_trained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, num_classes=1000)
            self.pre_trained_model.fc_2 = self.pre_trained_model.fc
            self.pre_trained_model.fc = torch.nn.Identity()
            free_model_output_dim = 512
        elif client_dict['pre_trained'] == 'eb0':
            self.pre_trained_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes=1000)
            self.pre_trained_model.fc_2 = self.pre_trained_model.classifier
            self.pre_trained_model.classifier = torch.nn.Identity()
            self.pre_trained_model.fc = self.pre_trained_model.classifier
            free_model_output_dim = 1280
        else:
            raise Exception('free lunch pre_trained_model not identified')

        for param in self.pre_trained_model.parameters():
            param.detach_()

        self.model = self.model_type(self.num_classes, KD=True, projection=True,
                                     free_model_output_dim=free_model_output_dim)
        self.prev_model = self.model_type(self.num_classes, KD=True, projection=True,
                                          free_model_output_dim=free_model_output_dim)

        self.global_model = self.model_type(self.num_classes, KD=True, projection=True,
                                            free_model_output_dim=free_model_output_dim)
        self.feat_dim = self.model.fc.in_features
        self.criterion_dict = client_dict['criterion']
        self.criterion_bal_l2_dict = client_dict['criterion_bal_l2']

        self.free_criterion_dict = client_dict['free_criterion']

        #
        self.model.to(self.device)
        if self.args.adam:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,
                                             weight_decay=self.args.wd, nesterov=True)

        self.optimizer_client_states = {i: self.optimizer.state_dict() for i in client_dict['client_map'][0]}

        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.temp = 0.5

        if self.args.loss_fn_name == 'LDAM':
            self.model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)
            self.model.fc_free = ClassifierLDAM(free_model_output_dim, self.num_classes, self.device)

            self.global_model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)
            self.prev_model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)
            self.global_model.fc_free = ClassifierLDAM(free_model_output_dim, self.num_classes, self.device)
            self.prev_model.fc_free = ClassifierLDAM(free_model_output_dim, self.num_classes, self.device)

    def load_client_state_dict(self, server_state_dict):
        # If you want to customize how to state dict is loaded you can do so here
        self.model.load_state_dict(server_state_dict)
        if self.args.update_projector_head == 'free':
            self.model.fc.load_state_dict(self.prev_model.fc.state_dict())
            self.model.p.load_state_dict(self.prev_model.p.state_dict())
        elif self.args.update_projector_head == 'global':
            self.model.fc_free.load_state_dict(self.prev_model.fc_free.state_dict())
            self.model.p_free.load_state_dict(self.prev_model.p_free.state_dict())
        elif self.args.update_projector_head == 'both':
            pass
        else:
            raise ValueError("Update projector head not implemented yet")

        return

    def run(self, received_info):
        client_results = []
        for client_idx in self.client_map[self.round]:
            self.global_model.load_state_dict(received_info['global'])
            if self.args.update_clients == 1:
                self.load_client_state_dict(received_info['global'])

            self.train_dataloader = self.train_data[client_idx]

            self.val_dataloader = self.val_data[client_idx]
            self.test_dataloader = self.test_data[client_idx]

            self.class_compose = torch.tensor(np.array(self.get_class_compose())).to(self.device)

            self.criterion = self.criterion_dict[client_idx]
            if self.args.free_u == 0:
                self.criterion.reduction = 'none'

            self.criterion_free = self.free_criterion_dict[client_idx]
            self.criterion_bal_l2 = self.criterion_bal_l2_dict[client_idx]
            self.criterion_bal_l2.to(self.device)
            self.criterion.to(self.device)
            self.criterion_free.to(self.device)

            self.class_priors = self.get_class_priors()
            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None and self.train_dataloader._iterator._shutdown:
                self.train_dataloader._iterator = self.train_dataloader._get_iterator()

            self.client_index = client_idx
            num_samples = len(self.train_dataloader) * self.args.batch_size

            self.class_weights = torch.zeros_like(self.class_compose, dtype=torch.float).to(self.device)
            if self.args.aggregate_method != 'avg':
                if not (self.args.aggregate_method == 'higher_mix_DRW' and self.round < int(
                        self.args.comm_round * 0.8)):
                    with torch.no_grad():
                        self.get_class_weights()
                    logger.info(f'calculating client_class_weights {self.class_weights}')

            weights, total_loss_dict = self.train()

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

            client_results.append(
                {'weights': copy.deepcopy(weights),
                 'num_samples': num_samples,
                 'acc': acc,
                 'client_index': self.client_index,
                 'class_weights': copy.deepcopy(self.class_weights.cpu()),
                 'loss': total_loss_dict['loss']
                    , 'loss_sup': total_loss_dict['loss_sup'], 'loss_glob': total_loss_dict['loss_glob'],
                 'loss_free': total_loss_dict['loss_free'],
                 'class_losses': copy.deepcopy(total_loss_dict['class_losses']),
                 'robust_factor': copy.deepcopy(total_loss_dict['robust_factor'])})

            if self.args.client_sample < 1.0 and self.train_dataloader._iterator is not None:
                self.train_dataloader._iterator._shutdown_workers()

        self.round += 1
        return client_results

    def get_class_weights(self):
        self.model.eval()
        self.model.to(self.device)

        self.pre_trained_model.to(self.device)
        if self.args.loss_fn_name == 'LDAM':
            self.criterion.set_weight(self.round)

        for batch_idx, (x, target) in enumerate(self.train_dataloader):
            # logger.info(x.shape)
            x, target = x.to(self.device), target.to(self.device).long()
            proj_1, out_1, proj_2, out_2 = self.model(x)

            if self.args.free_u != 0:
                if self.pre_Trained_model_type == 'clip':
                    pass
                    # free_inputs = self.normalizer(x, reverse=True)
                    # x_free = self.pre_trained_transform(free_inputs)
                    # x_free = x_free.to(self.device)
                    # proj_2_free = self.pre_trained_model.encode_image(x_free)

                elif self.pre_Trained_model_type == 'moco' \
                        or self.pre_Trained_model_type == 'IMAGENET1K_V1' \
                        or self.pre_Trained_model_type == '18_img' \
                        or self.pre_Trained_model_type == 'dinov2' \
                        or self.pre_Trained_model_type == 'eb0':

                    proj_2_free = self.pre_trained_model(x)

                loss_free = self.l2_loss_fn(proj_2, proj_2_free)
            else:
                loss_free = self.criterion(out_1, target).float()

            unique_labels, labels_count = target.unique(return_counts=True)
            unique_labels = unique_labels.long()

            res = torch.zeros_like(self.class_weights, dtype=torch.float).scatter_add_(0, target.type(torch.int64),
                                                                                       loss_free.detach().clone())
            res[unique_labels] = res[unique_labels] / labels_count.float()
            target_class_total_count = self.class_compose[unique_labels]
            self.class_weights[unique_labels] += (labels_count / target_class_total_count) * \
                                                 res[unique_labels]

    def l2_loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def get_class_compose(self):
        class_count_per_client = [0] * (self.num_classes)
        class_idx, class_count_per_client_tr = np.unique(self.train_dataloader.dataset.target,
                                                         return_counts=True)
        for idx, j in enumerate(class_idx):
            class_count_per_client[j] = class_count_per_client_tr[idx]
        return class_count_per_client

    def train(self):

        # train the local model
        self.model.to(self.device)
        self.global_model.to(self.device)
        self.pre_trained_model.to(self.device)
        self.prev_model.to(self.device)
        self.optimizer.load_state_dict(self.optimizer_client_states[self.client_index])

        self.model.train()
        self.pre_trained_model.eval()

        if self.args.adjust_learning_rate:
            adjust_learning_rate(self.optimizer, self.round, self.args.lr, self.args.comm_round,
                                 multi_step=self.args.multi_step)

        if self.args.balanced_free:
            self.criterion_bal_l2.set_weight(self.round)


        epoch_loss = []
        epoch_loss_sup = []
        epoch_loss_glob = []
        epoch_loss_free = []
        ramp_up_kl_glob = 1
        ramp_down_kl = 1
        if self.args.loss_fn_name == 'LDAM':
            self.criterion.set_weight(self.round)
            self.criterion_free.set_weight(self.round)

        class_losses = torch.zeros_like(self.class_compose, dtype=torch.float).to(self.device)
        for epoch in range(self.args.epochs):
            batch_loss = []
            batch_loss_sup = []
            batch_loss_glob = []
            batch_loss_free = []

            for batch_idx, (x, target) in enumerate(self.train_dataloader):
                # logger.info(x.shape)
                x, target = x.to(self.device), target.to(self.device).long()
                self.optimizer.zero_grad()
                proj_1, out_1, proj_2, out_2 = self.model(x)
                loss_glob = torch.tensor(0)

                # x without prompt tuning
                free_inputs = x

                if self.pre_Trained_model_type == 'clip':
                    pass
                    # free_inputs = self.normalizer(free_inputs, reverse=True)
                    # x_free = self.pre_trained_transform(free_inputs)
                    # x_free = x_free.to(self.device)
                    # proj_2_free = self.pre_trained_model.encode_image(x_free)
                elif self.pre_Trained_model_type == 'moco' \
                        or self.pre_Trained_model_type == 'IMAGENET1K_V1' \
                        or self.pre_Trained_model_type == '18_img' \
                        or self.pre_Trained_model_type == 'dinov2' \
                        or self.pre_Trained_model_type == 'eb0':
                    proj_2_free = self.pre_trained_model(free_inputs)

                loss_sup = self.criterion(out_1, target)

                loss_sup_2 = self.criterion_free(out_2, target)


                if self.args.balanced_free:
                    loss_free = self.criterion_bal_l2(proj_2, proj_2_free, target)
                else:
                    loss_free = self.l2_loss_fn(proj_2, proj_2_free)

                if self.args.free_u == 0:
                    loss_free = loss_sup.detach().float()
                    loss_sup = loss_sup.mean()

                #ramping down the free factor does not enhance the accuraccy
                #due to the fact that the MLP projector needs to be trained to align all clients to the reference distribution
                if self.args.LRD:
                    ramp_up_kl_glob = self.ramp_up(current=self.round)
                    if self.round >= self.args.start_free_rd:
                        ramp_down_kl = 1 - self.ramp_up_free(current=self.round - self.args.start_free_rd)

                self.free_factor = self.args.free_u * ramp_down_kl
                self.distill_factor = self.args.distill_u * ramp_up_kl_glob

                loss = self.distill_factor * loss_glob + self.free_factor * loss_free.mean() + loss_sup + loss_sup_2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()


                # calculate the class-aware average mean
                # parallelized with torch
                unique_labels, labels_count = target.unique(return_counts=True)
                unique_labels = unique_labels.long()
                res = torch.zeros_like(class_losses, dtype=torch.float).scatter_add_(0, target.type(torch.int64),
                                                                                     loss_free.detach().clone())
                res[unique_labels] = res[unique_labels] / labels_count.float()
                target_class_total_count = self.class_compose[unique_labels]
                class_losses[unique_labels] += (labels_count / target_class_total_count) * \
                                               res[unique_labels]



                #for plotting
                batch_loss.append(loss.item())
                batch_loss_sup.append(loss_sup.item())
                batch_loss_glob.append(loss_glob.item())
                batch_loss_free.append(loss_free.detach().mean().item())

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                epoch_loss_sup.append(sum(batch_loss_sup) / len(batch_loss_sup))
                epoch_loss_glob.append(sum(batch_loss_glob) / len(batch_loss_glob))
                epoch_loss_free.append(sum(batch_loss_free) / len(batch_loss_free))
                total_loss = sum(
                    epoch_loss) / len(epoch_loss)
                total_loss_sup = sum(
                    epoch_loss_sup) / len(epoch_loss_sup)
                total_loss_glob = sum(
                    epoch_loss_glob) / len(epoch_loss_glob)
                total_loss_free = sum(
                    epoch_loss_free) / len(epoch_loss_free)
                total_loss_dict = {'loss': total_loss, 'loss_sup': total_loss_sup, 'loss_glob': total_loss_glob,
                                   'loss_free': total_loss_free}
                logger.info(
                    '(client {}. Local Training Epoch: {} \tLoss: {:.6f}, Loss_sup: {:.6f}, Loss_glob: {:.6f}, Loss_free: {:.6f}, Free_Factor {}, Distill_Factor:{},  Thread {}  Map {}'.format(
                        self.client_index,
                        epoch, total_loss, total_loss_sup, total_loss_glob, total_loss_free, self.free_factor,
                        self.distill_factor,
                        current_process()._identity[0],
                        self.client_map[self.round]))

        if self.args.ramp_fed:
            self.free_factor = min(self.args.free_u,
                                   total_loss_sup /
                                   (self.args.free_u * total_loss_free))

        total_loss_dict['class_losses'] = class_losses.cpu()

        if not self.args.aggregate_method == 'avg':
            non_zero_idx = self.class_weights != 0

            if self.args.aggregate_method == 'mas_inv':
                class_losses[non_zero_idx] = 1 / class_losses[non_zero_idx]

            class_weights = self.class_weights / torch.sum(self.class_weights)

            class_losses_norm = copy.deepcopy(class_losses)
            class_losses_norm[non_zero_idx] = class_losses[non_zero_idx] / torch.sum(class_losses)

            total_loss_dict['robust_factor'] = torch.sum(
                (class_weights[non_zero_idx]) * class_losses_norm[non_zero_idx]).item()
        else:
            total_loss_dict['robust_factor'] = 0

        self.optimizer_client_states[self.client_index] = copy.deepcopy(self.optimizer.state_dict())

        weights = self.model.cpu().state_dict()
        return weights, total_loss_dict

    def test(self):
        if self.args.validate_client and self.round == self.args.comm_round - 1:
            logger.info(f'Loading Best Model for Client {self.client_index}')
            save_client = True
            self.model.load_state_dict(self.client_best_weights[self.client_index])
            test_data = self.val_dataloader
        else:
            save_client = False
            test_data = self.test_dataloader

        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_sample_number = 0.0
        test_correct_free = 0.0
        pred_list = []
        target_list = []
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                _, out_1, _, _ = self.model(x)

                _, predicted = torch.max(out_1, 1)

                correct = predicted.eq(target).sum()

                test_correct += correct.item()

                pred_list.extend(predicted.cpu().detach().squeeze().tolist())
                target_list.extend(target.cpu().detach().squeeze().tolist())

                # test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            acc = (test_correct / test_sample_number) * 100
            acc_free = (test_correct_free / test_sample_number) * 100

            logger.info("************* Client {} Acc = {:.2f} **************".format(self.client_index, acc, acc_free))

        if save_client:
            d = {'pred': pred_list, 'target': target_list}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(self.test_file_path, f'{self.client_index}.csv'))

        acc = balanced_accuracy_score(target_list, pred_list)
        return acc


class Server(Base_Server):
    def __init__(self, server_dict, args):
        super().__init__(server_dict, args)
        if args.pre_trained_models == 'mix':
            free_model_output_dim = 2048
        elif args.pre_trained_models == 'moco' or args.pre_trained_models == 'IMAGENET1K_V1':
            free_model_output_dim = 2048
        elif args.pre_trained_models == 'clip' or args.pre_trained_models == '18_img':
            free_model_output_dim = 512
        elif args.pre_trained_models == 'dinov2':
            free_model_output_dim = 384
        elif args.pre_trained_models == 'eb0':
            free_model_output_dim = 1280
        else:
            logger.info('pre_trained _models not defined')
            raise Exception
        self.model = self.model_type(self.num_classes, KD=True, projection=True,
                                     free_model_output_dim=free_model_output_dim)

        if self.args.pre_trained_tv:
            new_state_dict = get_pre_trained_model_dict(args, self.model.state_dict())
            self.model.load_state_dict(new_state_dict)

        self.feat_dim = self.model.fc.in_features
        if self.args.loss_fn_name == 'LDAM':
            self.model.fc = ClassifierLDAM(self.feat_dim, self.num_classes, self.device)
            self.model.fc_free = ClassifierLDAM(free_model_output_dim, self.num_classes, self.device)

        self.best_weights = copy.deepcopy(self.model.state_dict())

    def operations(self, client_info):
        client_info.sort(key=lambda tup: tup['client_index'])
        client_sd = [c['weights'] for c in client_info]
        if self.args.aggregate_method == "avg":
            cw = [c['num_samples'] / sum([x['num_samples'] for x in client_info]) for c in client_info]
            logger.info('AVG server aggregation')
        elif self.args.aggregate_method == "mas":
            cw = [c['robust_factor'] / sum([x['robust_factor'] for x in client_info]) for c in client_info]
            logger.info('higher loss have higher weights')
        elif self.args.aggregate_method == "mas_inv":
            cw = [c['robust_factor'] / sum([x['robust_factor'] for x in client_info]) for c in client_info]
            logger.info('higher loss free have higher weights but far from the model')
        #delayed Re-weighitng to prevent overfitting miniority classes
        #extract representations from the majority followed by reweighitng to capture local clients variations
        elif self.args.aggregate_method == "mas_DRW":
            if self.round >= int(self.args.comm_round * 0.8):
                cw1 = [(1 / c['robust_factor']) / sum([1 / x['robust_factor'] for x in client_info]) for c in
                       client_info]
                idx = self.round // int(self.args.comm_round * 0.8)
            else:
                cw1 = [1 / sum([1 for x in client_info]) for c in client_info]
                idx = 0
            cls_num_list = np.array(cw1)
            non_zero_indices = cls_num_list != 0
            betas = [0, 0.9999]
            effective_num = np.zeros_like(cls_num_list, dtype=np.float32)
            effective_num[non_zero_indices] = 1.0 - np.power(betas[idx], cls_num_list[non_zero_indices])
            per_cls_weights = np.zeros_like(cls_num_list, dtype=np.float32)
            per_cls_weights[non_zero_indices] = (1.0 - betas[idx]) / np.array(effective_num[non_zero_indices])
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list[non_zero_indices])

            cw = [c['num_samples'] * per_cls_weights[j] / sum(
                [x['num_samples'] * per_cls_weights[i] for i, x in enumerate(client_info)]) for j, c in
                  enumerate(client_info)]
            logger.info(f'DRW weights,{idx, per_cls_weights}')

        ssd = self.model.state_dict()

        #updating the global model with the rescue factor
        # may also be updated with EMA.
        for key in ssd:
            ssd[key] = sum([sd[key] * cw[i] for i, sd in enumerate(client_sd)])

        self.model.load_state_dict(ssd)

        if self.args.save_client:
            for client in client_info:
                torch.save(client['weights'], '{}/client_{}.pt'.format(self.save_path, client['client_index']))
        return [{'weights': self.model.cpu().state_dict()} for
                x in range(self.args.thread_number)]

    def run(self, received_info):
        server_outputs = self.operations(received_info)
        validate = False
        if self.round == self.args.comm_round - 1:
            self.model.load_state_dict(self.best_weights)
            logger.info("************* Loading best weights For Test Set of EyePacs **************")
            validate = True

        acc = self.test(validate=validate)

        logger.info("************* Server Acc = {:.2f} **************".format(acc))

        self.log_info(received_info, acc)
        self.round += 1
        if acc > self.acc:
            logger.info('Best weights achieved loading best weights')
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.acc = acc

        server_outputs = [{'global': g['weights'],
                           'server_acc': acc
                           } for g
                          in server_outputs]
        return server_outputs

    def start(self):
        return [{'global': self.model.cpu().state_dict(), 'server_acc': 0} for x in
                range(self.args.thread_number)]

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

                _, out_1, _, _ = self.model(x)

                _, predicted = torch.max(out_1, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()

                pred_list.extend(predicted.cpu().detach().squeeze().tolist())

                target_list.extend(target.cpu().detach().squeeze().tolist())

                test_sample_number += target.size(0)

        if validate:
            d = {'pred': pred_list, 'target': target_list}
            df = pd.DataFrame(d)
            df.to_csv(os.path.join(self.test_file_path, 'server.csv'))

        acc = balanced_accuracy_score(target_list, pred_list)

        return acc
