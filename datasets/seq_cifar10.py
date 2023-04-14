# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path_dataset as base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from torch.utils.data import DataLoader, Dataset, TensorDataset
import kornia as K
import torch

class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    
    train_transform = torch.nn.Sequential(
                K.augmentation.RandomCrop((32, 32), padding=4, same_on_batch=False),
                K.augmentation.RandomHorizontalFlip(same_on_batch=False),
                K.augmentation.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2470, 0.2435, 0.2615)),
            )
    test_transform = torch.nn.Sequential(
                K.augmentation.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2470, 0.2435, 0.2615)),
            )
    
    train_set=CIFAR10(base_path() + 'CIFAR10',train=True,download=True)
    test_set=CIFAR10(base_path() + 'CIFAR10',train=False,download=True)
    train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
    test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

    train_data = train_data.permute(0, 3, 1, 2)/255.0
    test_data = test_data.permute(0, 3, 1, 2)/255.0

    def get_data_loaders(self):
        train_mask = (self.train_targets >= self.i) & (self.train_targets < self.i + self.N_CLASSES_PER_TASK)
        test_mask = (self.test_targets >= self.i) & (self.test_targets < self.i + self.N_CLASSES_PER_TASK)

        train_loader = DataLoader(TensorDataset(self.train_data[train_mask], self.train_targets[train_mask]), batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(self.test_data[test_mask], self.test_targets[test_mask]), batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.i += self.N_CLASSES_PER_TASK
        return train_loader, test_loader

    @staticmethod
    def get_transform():
        return SequentialCIFAR10.train_transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return SequentialCIFAR10.test_transform 

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()
