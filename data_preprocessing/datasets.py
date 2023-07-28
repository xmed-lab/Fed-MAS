'''
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
'''

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision
from PIL import Image
from torchvision.datasets import CIFAR10, SVHN
from torchvision.datasets import CIFAR100
from torchvision.datasets import DatasetFolder, ImageFolder

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



class COVID19_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, return_index=False, fold=None, cache=False, imb_factor=None, test_set=False):
        self.return_index = return_index
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        csv_path = None
        if train:
            csv_path = os.path.join(self.root, f'train_covid{fold}_{imb_factor}.csv')
            wrap_img_data = 'train'
        else:
            # server
            wrap_img_data = 'test'
            if test_set:
                csv_path = os.path.join(self.root, f'val_covid{fold}_{imb_factor}.csv')
            else:
                csv_path = os.path.join(self.root, f'test_covid{fold}_{imb_factor}.csv')

        if imb_factor != 'all':
            wrap_img_data = 'train'

        df = pd.read_csv(csv_path)
        im_list = df.image_id

        im_list = os.path.join(self.root, wrap_img_data) + '/' + im_list
        target = df[df.columns[-1]].values
        self.loader = self.pil_loader
        self.cache = cache

        im_list = np.array(im_list)
        target = np.array(target)
        if self.dataidxs is not None:
            im_list = im_list[self.dataidxs]
            target = target[self.dataidxs]


        self.target = target.astype(np.int64)
        self.samples = im_list

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.target[index]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return sample, target, index
        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)



class ISIC_FL_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, return_index=False, fold=None, cache=False, imb_factor=None, test_set=False):
        self.return_index = return_index
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        csv_path = None
        if train:
            csv_path = os.path.join(self.root, f'train_isic192_fl_{imb_factor}_{fold}.csv')
        else:
            # server
            if test_set:
                csv_path = os.path.join(self.root, f'val_isic192_fl_{imb_factor}_{fold}.csv')
            else:
                csv_path = os.path.join(self.root, f'test_isic192_fl_{imb_factor}_{fold}.csv')


        df = pd.read_csv(csv_path)

        im_list = df.image_name
        # im_list = self.root + '/' + im_list + '.jpg'
        im_list = 'data/ham2019' + '/' + im_list + '.jpg'
        target = df.dx.values

        self.loader = self.pil_loader
        self.cache = cache

        im_list = np.array(im_list)
        target = np.array(target)
        if self.dataidxs is not None:
            im_list = im_list[self.dataidxs]
            target = target[self.dataidxs]

        if cache:
            im_list = np.array([np.array(self.loader(fname)) for fname in im_list])

        self.target = target.astype(np.int64)
        self.samples = im_list
    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.target[index]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return sample, target, index
        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)



class Flamby_isic(ISIC_FL_custom):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False, return_index=False, fold=None, cache=False, imb_factor=None, test_set=False):
        self.return_index = return_index
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if train:
            csv_path = os.path.join(self.root, f'train_flamby_isic_fl_{imb_factor}_{fold}.csv')
        else:
            csv_path = os.path.join(self.root, f'val_flamby_isic_fl_{imb_factor}_{fold}.csv')

        df = pd.read_csv(csv_path)

        im_list = df.image_name
        im_list = 'data/ISIC_2019_Training_Input' + '/' + im_list + '.jpg'
        target = df.dx.values

        self.loader = self.pil_loader
        self.cache = cache

        im_list = np.array(im_list)
        target = np.array(target)
        if self.dataidxs is not None:
            im_list = im_list[self.dataidxs]
            target = target[self.dataidxs]

        if cache:
            im_list = np.array([np.array(self.loader(fname)) for fname in im_list])

        self.target = target.astype(np.int64)
        self.samples = im_list