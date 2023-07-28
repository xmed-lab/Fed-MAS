import copy

import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights
import torch


def get_pre_trained_model_dict(args, new_state_dict):
    pre_trained_weights = None
    if args.model == 'resnet18':
        pre_trained_weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    elif args.model == 'effnetb0':
        pre_trained_weights = EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    elif args.pre_trained_models == 'moco':
        checkpoint = torch.load('./data/moco_v2_800ep_pretrain.pth.tar')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q.'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        pre_trained_temp = resnet50(weights=None, num_classes=128)
        dim_mlp = pre_trained_temp.fc.weight.shape[1]
        pre_trained_temp.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), pre_trained_temp.fc)
        pre_trained_temp.load_state_dict(state_dict)
        pre_trained_temp.fc = torch.nn.Identity()
        pre_trained_weights = pre_trained_temp.state_dict()
        # logger.info('MoCoV2 pre-trained model loaded successfuly')
    elif args.pre_trained_models == 'IMAGENET1K_V1':
        # getting imagenet pretrained weights
        if args.model == 'resnet18':
            pre_trained_weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        elif args.model == 'resnet50':
            pre_trained_weights = ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        else:
            # logger.info('pre_trained _models loaded defined')
            raise Exception
        # logger.info('ImageNet pre-trained model loaded successfuly')
    elif args.pre_trained_models == 'radImageNet':
        # RadImageNet - ResNet50_notop.h5
        pre_trained_weights = torch.load('./data/RadImageNet-ResNet50_notop_torch.pth')
    else:
        # logger.info('pre_trained _models loaded defined')
        raise Exception
    # we do not load the fc of the pre-trained weight
    for k, v in pre_trained_weights.items():
        if any(sub_x in k for sub_x in ['fc', 'p1', 'p2']):
            pass
        else:
            if new_state_dict.get(k) is not None:
                new_state_dict[k] = v
            else:
                print(k)
                raise Exception
    return new_state_dict


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'  # 'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5  # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def DiffAugment(x, strategy='', seed=-1, param=None):
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M':  # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s' % param.aug_mode)
        x = x.contiguous()
    return x


def rand_scale(x, param):
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0, 0],
              [0, sy[i], 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param):
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:  # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1




def match_loss(gw_syn, gw_real, device, dis_metric):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        # return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


class BlSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction="mean"):
        super(BlSoftmaxLoss, self).__init__()
        # reduction: string. One of "none", "mean", "sum"
        label_count_array = cls_num_list
        label_count_array = np.array(label_count_array) / np.sum(label_count_array)
        adjustments = np.log(label_count_array + 1e-12)
        adjustments = torch.from_numpy(adjustments).view(1, -1)
        self.adjustments = adjustments
        self.reduction = reduction

    def forward(self, logits, target):
        logits = logits + self.adjustments.to(logits.device)
        loss = F.cross_entropy(input=logits, target=target, reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            assert False

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def count_dataset(train_loader):
    label_freq = {}
    if isinstance(train_loader, list) or isinstance(train_loader, tuple):
        all_labels = train_loader[0].dataset.target
    else:
        all_labels = train_loader.dataset.target
    for label in all_labels:
        key = str(label)
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    return label_freq_array


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, total_epoch, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        self.cls_num_list = np.array(cls_num_list)
        self.non_zero_indices = self.cls_num_list != 0
        m_list = np.zeros_like(self.cls_num_list, dtype=np.float32)
        m_list[self.non_zero_indices != 0] = 1.0 / np.sqrt(np.sqrt(self.cls_num_list[self.non_zero_indices]))
        m_list[self.non_zero_indices] = m_list[self.non_zero_indices] * (max_m / np.max(m_list[self.non_zero_indices]))

        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.total_epoch = total_epoch

    def set_weight(self, epoch):
        idx = epoch // int(self.total_epoch * 0.8)
        betas = [0, 0.9999]
        effective_num = np.zeros_like(self.cls_num_list, dtype=np.float32)
        effective_num[self.non_zero_indices] = 1.0 - np.power(betas[idx], self.cls_num_list[self.non_zero_indices])
        per_cls_weights = np.zeros_like(self.cls_num_list, dtype=np.float32)
        per_cls_weights[self.non_zero_indices] = (1.0 - betas[idx]) / np.array(effective_num[self.non_zero_indices])
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list[self.non_zero_indices])
        self.weight = torch.FloatTensor(per_cls_weights)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float().to(x.device)
        batch_m = torch.matmul(self.m_list.to(x.device)[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight.to(x.device))


def create_criterion_dict(train_data_local_dict, rounds, loss_fn_name, num_classes, byol=False):
    criterion_dict = dict()
    criterion_dict_bal_l2 = dict()
    for i in range(len(train_data_local_dict)):
        class_count_per_client = [0] * (num_classes)
        class_idx, class_count_per_client_tr = np.unique(train_data_local_dict[i].dataset.target, return_counts=True)
        for idx, j in enumerate(class_idx):
            class_count_per_client[j] = class_count_per_client_tr[idx]
        criterion_dict[i] = get_criterion(class_count_per_client, rounds, loss_fn_name)
        criterion_dict_bal_l2[i] = get_criterion(class_count_per_client, rounds, 'balanced_l2')
    if byol:
        return criterion_dict, criterion_dict_bal_l2
    return criterion_dict

def get_criterion(cls_num_list, rounds, loss_fn_name):
    if loss_fn_name == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_fn_name == 'BSM':
        loss_fn = BlSoftmaxLoss(cls_num_list)
    elif loss_fn_name == 'focal':
        loss_fn = FocalLoss(gamma=2.0)
    elif loss_fn_name == 'LDAM':
        loss_fn = LDAMLoss(cls_num_list, total_epoch=rounds)
    elif loss_fn_name == 'CB':
        cls_num_list = np.array(cls_num_list)
        non_zero_indices = cls_num_list != 0
        beta = 0.9999
        effective_num = np.zeros_like(cls_num_list, dtype=np.float32)
        effective_num[non_zero_indices] = 1.0 - np.power(beta, cls_num_list[non_zero_indices])
        per_cls_weights = np.zeros_like(cls_num_list, dtype=np.float32)
        per_cls_weights[non_zero_indices] = (1.0 - beta) / np.array(effective_num[non_zero_indices])
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list[non_zero_indices])
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        loss_fn = nn.CrossEntropyLoss(weight=per_cls_weights)
    elif loss_fn_name == 'balanced_l2':
        loss_fn = BalancedSupByolLoss(cls_num_list, total_epoch=rounds)
    elif loss_fn_name == 'RW':
        cls_num_list = np.array(cls_num_list)
        non_zero_indices = cls_num_list != 0
        weights = np.zeros_like(cls_num_list, dtype=np.float32)
        weights[non_zero_indices] = np.sum(cls_num_list) / (len(cls_num_list) * (cls_num_list[non_zero_indices]))
        weights = torch.FloatTensor(weights)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    elif loss_fn_name == 'LCC':
        cls_num_list = np.array(cls_num_list)
        loss_fn = LCC(cls_num_list=cls_num_list, T=0.5)

    else:

        raise Exception("this loss is not defined")

    return loss_fn


class LCC(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T):
        super(LCC, self).__init__()
        self.T = T
        self.label_distrib = torch.zeros(len(cls_num_list))
        for cls, count in enumerate(cls_num_list):
            self.label_distrib[cls] = max(1e-8, count)

    def forward(self, logit, y):
        tmp = torch.pow(self.label_distrib, -1 / 4).unsqueeze(0).expand((logit.shape[0], -1)).to(logit.device)
        cal_logit = torch.exp(
            logit
            - (
                    self.T
                    * tmp
            )
        )

        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))

        return loss.sum() / logit.shape[0]



class ClassifierLDAM(nn.Module):
    def __init__(self, feat_dim, num_classes=1000, device=None):
        super(ClassifierLDAM, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes).to(device), requires_grad=True)
        self.weight.data.uniform_(-1, 1)
        self.weight.data.renorm_(2, 1, 1e-5)
        self.weight.data.mul_(1e5)

    def forward(self, x, add_inputs=None):
        y = torch.mm(F.normalize(x, dim=1), F.normalize(self.weight, dim=0))
        return y


class BalancedSupByolLoss(nn.Module):
    def __init__(self, cls_num_list, total_epoch, ratio=0.8):
        super(BalancedSupByolLoss, self).__init__()
        self.ratio = ratio
        self.cls_num_list = np.array(cls_num_list)
        self.non_zero_indices = self.cls_num_list != 0
        self.total_epoch = total_epoch
        self.m = torch.Tensor(self.cls_num_list / np.sum(cls_num_list))

    def set_weight(self, epoch):
        idx = epoch // int(self.total_epoch * self.ratio)
        betas = [0, 0.9999]
        effective_num = np.zeros_like(self.cls_num_list, dtype=np.float32)
        effective_num[self.non_zero_indices] = 1.0 - np.power(betas[idx], self.cls_num_list[self.non_zero_indices])
        per_cls_weights = np.zeros_like(self.cls_num_list, dtype=np.float32)
        per_cls_weights[self.non_zero_indices] = (1.0 - betas[idx]) / np.array(effective_num[self.non_zero_indices])
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list[self.non_zero_indices])
        self.weight = torch.FloatTensor(per_cls_weights)

    def forward(self, x, y, target):
        index = torch.zeros((x.size(0), self.weight.size(0)), dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.weight.to(x.device)[None, :], index_float.transpose(0, 1))
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return (2 - 2 * (x * y).sum(dim=-1)) * batch_m



class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        self.centers.to(device)

    def forward(self, x, labels):
        center = self.centers[labels]
        dist = (x - center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterCosLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterCosLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def forward(self, x, labels):
        center = self.centers[labels]
        norm_c = self.l2_norm(center)
        norm_x = self.l2_norm(x)
        similarity = (norm_c * norm_x).sum(dim=-1)
        dist = 1.0 - similarity
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss


class CenterTripletLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterTripletLoss, self).__init__()
        self.num_class = num_classes
        self.num_feature = feat_dim
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, x, preds, labels):
        # use most likely categories as negative samples
        preds = preds.softmax(-1)
        batch_size = x.shape[0]
        idxs = torch.arange(batch_size).to(x.device)
        preds[idxs, labels] = -1
        adv_labels = preds.max(-1)[1]

        anchor = x  # num_batch, num_dim
        positive = self.centers[labels]  # num_batch, num_dim
        negative = self.centers[adv_labels]  # num_batch, num_dim

        output = self.triplet_loss(anchor, positive, negative)
        return output


def compute_lr(current_round, rounds=200, eta_min=0, eta_max=0.3):
    """Compute learning rate as cosine decay"""
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * current_round / rounds) + 1)
    return eta_t


def adjust_learning_rate(optimizer, epoch, lr, total_rounds=200, multi_step=False):
    """Decay the learning rate based on schedule"""
    # lr *= 0.5 * (1. + math.cos(math.pi * epoch / (self.args.rounds)))
    if multi_step:
        if epoch < 60:
            lr = optimizer.param_groups[0]['lr']
        elif 60 <= epoch < 70:
            lr = optimizer.param_groups[0]['lr'] * 0.1
        else:
            lr = optimizer.param_groups[0]['lr'] * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        epoch = epoch + 1
        lr = compute_lr(epoch, total_rounds, 1e-4, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_effective_number_weight(cls_num_list):
    betas = 0.9999
    cls_num_list = np.array(cls_num_list)
    non_zero_indices = cls_num_list != 0
    effective_num = np.zeros_like(cls_num_list, dtype=np.float32)
    effective_num[non_zero_indices] = 1.0 - np.power(betas, cls_num_list[non_zero_indices])
    per_cls_weights = np.zeros_like(cls_num_list, dtype=np.float32)
    per_cls_weights[non_zero_indices] = (1.0 - betas) / np.array(effective_num[non_zero_indices])
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list[non_zero_indices])
    weight = torch.FloatTensor(per_cls_weights)
    return weight


# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches
