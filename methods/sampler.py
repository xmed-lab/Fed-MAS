import random

import pandas as pd
import torch
import numpy as np
from torch.utils.data.sampler import Sampler


class WeightedSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float()  # init weight

    def __iter__(self):
        selected_inds = []
        # MAKE SURE self.weight.sum() == self.num_samples
        while ((self.weight >= 1.0).sum().item() > 0):
            inds = self.indexes[self.weight >= 1.0].tolist()
            selected_inds = selected_inds + inds
            self.weight = self.weight - 1.0
        selected_inds = torch.LongTensor(selected_inds)
        # shuffle
        current_size = selected_inds.shape[0]
        selected_inds = selected_inds[torch.randperm(current_size)]
        expand = torch.randperm(self.num_samples) % current_size
        indices = selected_inds[expand].tolist()

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()


class DistributionSampler(Sampler):
    def __init__(self, dataset):
        self.num_samples = len(dataset)
        self.indexes = torch.arange(self.num_samples)
        self.weight = torch.zeros_like(self.indexes).fill_(1.0).float()  # init weight

    def __iter__(self):
        self.prob = self.weight / self.weight.sum()
        indices = torch.multinomial(self.prob, self.num_samples, replacement=True).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, weight):
        self.weight = weight.float()


class FixSeedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_parameter(self, epoch):
        self.epoch = epoch


import random
import numpy as np
from torch.utils.data.sampler import Sampler


##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
## github: https://github.com/facebookresearch/classifier-balancing/blob/main/data/ClassAwareSampler.py
##################################

class RandomCycleIter:

    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        random.shuffle(self.data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_classes, num_samples_cls=1, max_aug=1):
        num_images = len(data_source.target)
        unq = np.unique(data_source.target)
        num_classes = len(unq)
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = {label: list() for label in unq}
        for i, label in enumerate(data_source.target):
            cls_data_list[label].append(i)

        self.data_iter_list = [RandomCycleIter(x[1]) for x in cls_data_list.items()]
        self.num_samples = max([len(x[1]) for x in cls_data_list.items()]) * len(cls_data_list)
        if self.num_samples > (num_images * max_aug):
            self.num_samples = num_images * max_aug
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
            self,
            dataset,
            indices: list = None,
            num_samples: int = None,
            replacement=True,
            weights=None
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        self.replacement = replacement
        df = pd.DataFrame()
        df["label"] = dataset.target
        df.index = self.indices

        df = df.sort_index()
        label_to_count = df["label"].value_counts()

        # get the sampling weights
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())
        # self.weights = weights

    def __iter__(self):
        return (self.indices[i] for i in
                torch.multinomial(self.weights, self.num_samples, replacement=self.replacement))

    def __len__(self):
        return self.num_samples
