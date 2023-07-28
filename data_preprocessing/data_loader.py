'''
Federated Dataset Loading and Partitioning
Code based on https://github.com/FedML-AI/FedML
'''
import copy
import logging
import pickle
import sys
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.datasets import ISIC_FL_custom, Flamby_isic
from methods.sampler import WeightedSampler, ImbalancedDatasetSampler


import math
import functools
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    class_num_unqs = np.unique(y_train)
    class_num = len(class_num_unqs)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {i: 0 for i in range(class_num)}
        for i in range(len(unq)):
            tmp[unq[i]] = unq_cnt[i]
        net_cls_counts[net_i] = tmp
    logger.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts




def _data_transforms_flamby_isic(aug):
    #########################
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    input_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    input_transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return input_transform_train, input_transform_test


def _data_transforms_skin(aug=True):
    # transforms definition
    # required transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_resize = 224
    resize = transforms.Resize((image_resize, image_resize))
    tensorizer = transforms.ToTensor()
    # geometric transforms
    h_flip = transforms.RandomHorizontalFlip()
    v_flip = transforms.RandomVerticalFlip()
    rotate = transforms.RandomRotation(degrees=45)
    scale = transforms.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = transforms.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = transforms.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    train_transform = transforms.Compose(
        [resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer, transforms.Normalize(mean=mean, std=std)])
    valid_transform = transforms.Compose([resize, tensorizer, transforms.Normalize(mean=mean, std=std)])
    return train_transform, valid_transform


def calculate_reverse_instance_weight(dataset, num_class):
    # counting frequency
    label_freq = {}
    for i in range(num_class):
        label_freq[i] = 0
    for key in dataset.target:
        label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))

    # [50,40,30,20,10]
    # [,0.1,0.01,0.001]
    label_freq_array = torch.FloatTensor(list(label_freq.values()))
    reverse_class_weight = label_freq_array.max() / label_freq_array
    # generate reverse weight
    reverse_instance_weight = torch.zeros(len(dataset)).fill_(1.0)
    for i, label in enumerate(dataset.target):
        reverse_instance_weight[i] = reverse_class_weight[label] / (label_freq_array[label] + 1e-9)
    return reverse_instance_weight



def get_dataloader_covid(data_dir, batch_size, client_name, instance_sampling=None,
                         return_index=None,
                         num_class=None, fold=None, cache=None, aug=True):
    workers = 8
    persist = True

    if 'isic-FL' in data_dir:
        dl_obj = ISIC_FL_custom
        train_transform, test_transform = _data_transforms_skin(aug)
    elif 'Flamby_isic' in data_dir:
        dl_obj = Flamby_isic
        train_transform, test_transform = _data_transforms_flamby_isic(aug)
    else:
        raise Exception("data dir is not correct"
                        )
    train_ds = dl_obj(data_dir, train=True, download=True, transform=train_transform, fold=fold,
                      imb_factor=client_name)
    test_ds = dl_obj(data_dir, train=False, download=True, transform=test_transform, fold=fold,
                     imb_factor=client_name)
    val_ds = dl_obj(data_dir, train=False, download=True, test_set=True, transform=test_transform, fold=fold,
                    imb_factor=client_name)

    if instance_sampling:
        reverse_weight = calculate_reverse_instance_weight(train_ds, num_class)
        sampler = ImbalancedDatasetSampler(train_ds, weights=reverse_weight)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dl = data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                               num_workers=workers, sampler=sampler,
                               persistent_workers=persist)

    test_dl = data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=workers,
                              persistent_workers=persist)

    val_dl = data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=workers,
                             persistent_workers=persist)
    return train_dl, val_dl, test_dl


def load_partition_isic_fl(data_dir, client_number, batch_size,
                           instance_sampling=False, return_index=False, fold=None, cache=False, aug=True,
                           class_num=8):
    _, val_data_global, test_data_global = get_dataloader_covid(data_dir, batch_size, 'all',
                                                                instance_sampling=instance_sampling,
                                                                return_index=return_index, fold=fold,
                                                                cache=cache, aug=aug, num_class=class_num)
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        if client_number == 1:
            prompt = 'all'
        elif client_number==4:
            prompt = str(client_idx + 5)
        else:
            prompt = str(client_idx + 1)

        train_data_local, val_data_local, test_data_local = get_dataloader_covid(data_dir, batch_size, prompt,
                                                                                 instance_sampling=instance_sampling,
                                                                                 return_index=return_index, fold=fold,
                                                                                 cache=cache, aug=aug,
                                                                                 num_class=class_num)
        logger.info(f"client_idx = {client_idx}")
        logger.info(np.unique(train_data_local.dataset.target, return_counts=True))

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        val_data_local_dict[client_idx] = val_data_local

    return None, test_data_num, None, val_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, class_num


def load_partition_flampy_isic(data_dir, client_number, batch_size,
                               instance_sampling=False, return_index=False, fold=None, cache=False, aug=True,
                               class_num=8):

    if client_number == 1:
        client_idx_to_name = {0: 'all'}
    else:
        client_idx_to_name = {i: i for i in range(client_number)}


    _, val_data_global, test_data_global = get_dataloader_covid(data_dir, batch_size, 'all',
                                                                instance_sampling=instance_sampling,
                                                                return_index=return_index, fold=fold,
                                                                cache=cache, aug=aug, num_class=class_num)
    logger.info(np.unique(val_data_global.dataset.target, return_counts=True))
    logger.info(np.unique(test_data_global.dataset.target, return_counts=True))

    logger.info("val_dl_global number = " + str(len(val_data_global)))
    logger.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        client_name = client_idx_to_name[client_idx]

        train_data_local, val_data_local, test_data_local = get_dataloader_covid(data_dir, batch_size, client_name,
                                                                                 instance_sampling=instance_sampling,
                                                                                 return_index=return_index, fold=fold,
                                                                                 cache=cache, aug=aug,
                                                                                 num_class=class_num)
        logger.info(f"client_idx = {client_idx}")
        logger.info(np.unique(train_data_local.dataset.target, return_counts=True))

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        val_data_local_dict[client_idx] = val_data_local

    return None, test_data_num, None, val_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, class_num

def load_partition_COVID(data_dir, client_number, batch_size,
                         instance_sampling=False, return_index=False, fold=None, cache=False, aug=True,
                         class_num=3):
    if client_number == 1:
        client_idx_to_name = {0: 'all'}
    elif client_number == 5:
        client_idx_to_name = {
            0: "rsna-0",
            1: "cohen",
            2: "eurorad",
            3: "gz",
            4: "ml-workgroup",
            5: "ricord_c"}
    elif client_number == 8:
        client_idx_to_name = {
            0: "bimcv",
            1: "cohen",
            2: "eurorad",
            3: "gz",
            4: "ml-workgroup",
            5: "ricord_c",
            6: "rsna-0",
            7: "rsna-1",
            8: "sirm",

        }
    elif client_number == 12:
        client_idx_to_name = {
            0: "bimcv",
            1: "cohen",
            2: "rsna-0",
            3: "eurorad",
            4: "gz",
            5: "rsna-1",
            6: "ml-workgroup",
            7: "rsna-2",
            8: "ricord_c",
            9: "rsna-4",
            10: "sirm",
            11: "rsna-3"
        }
    else:
        raise Exception("Client n not supported")

    _, val_data_global, test_data_global = get_dataloader_covid(data_dir, batch_size, 'all',
                                                                instance_sampling=instance_sampling,
                                                                return_index=return_index, fold=fold,
                                                                cache=cache, aug=aug, num_class=class_num)
    logger.info(np.unique(val_data_global.dataset.target, return_counts=True))
    logger.info(np.unique(test_data_global.dataset.target, return_counts=True))

    logger.info("val_dl_global number = " + str(len(val_data_global)))
    logger.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        client_name = client_idx_to_name[client_idx]

        train_data_local, val_data_local, test_data_local = get_dataloader_covid(data_dir, batch_size, client_name,
                                                                                 instance_sampling=instance_sampling,
                                                                                 return_index=return_index, fold=fold,
                                                                                 cache=cache, aug=aug,
                                                                                 num_class=class_num)
        logger.info(f"client_idx = {client_idx}")
        logger.info(np.unique(train_data_local.dataset.target, return_counts=True))
        logger.info(np.unique(test_data_local.dataset.target, return_counts=True))
        logger.info(np.unique(val_data_local.dataset.target, return_counts=True))

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
        val_data_local_dict[client_idx] = val_data_local

    return None, test_data_num, None, val_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, class_num


