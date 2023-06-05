# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from torch.utils.data import DataLoader, Dataset, TensorDataset
import kornia as K
import torch


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 5
    INPUT_SHAPE = (3, 32, 32)

    train_transform = torch.nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), p=1, same_on_batch=False),
                K.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=False),
                K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=False),
                # K.augmentation.RandomGrayscale(p=0.2, same_on_batch=False),
            )
    
    test_transform = torch.nn.Sequential(
                K.augmentation.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761)),
            )
    test_transforms = []
    
    train_set=CIFAR100(base_path() + 'CIFAR100',train=True,download=True)
    test_set=CIFAR100(base_path() + 'CIFAR100',train=False,download=True)
    train_data, train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
    test_data, test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

    train_data = train_data.permute(0, 3, 1, 2)/255.0
    test_data = test_data.permute(0, 3, 1, 2)/255.0
    N_CLASSES = len(train_targets.unique())
    # print(train_data.mean((0, 2, 3)), train_data.std((0, 2, 3), unbiased=False))


    def get_data_loaders(self):
        train_mask = (self.train_targets >= self.i) & (self.train_targets < self.i + self.N_CLASSES_PER_TASK)
        test_mask = (self.test_targets >= self.i) & (self.test_targets < self.i + self.N_CLASSES_PER_TASK)

        train_loader = DataLoader(TensorDataset(self.train_data[train_mask], self.train_targets[train_mask]), batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(self.test_data[test_mask], self.test_targets[test_mask]), batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        if 'dis' in self.args.ablation:
            mean = train_loader.dataset.tensors[0].mean((0, 2, 3))
            std = train_loader.dataset.tensors[0].std((0, 2, 3), unbiased=False)
            print(f'Classes: {self.i} - {self.i+self.N_CLASSES_PER_TASK}, mean = {mean}, std = {std}')
            self.test_transforms += [torch.nn.Sequential(
                    K.augmentation.Normalize(mean, std)
                )]
        else:
            # self.test_transforms += [self.test_transform]
            self.test_transforms += [torch.nn.Sequential()]
        self.i += self.N_CLASSES_PER_TASK
        return train_loader, test_loader
    
    def get_full_data_loader(self):
        self.N_CLASSES_PER_TASK = self.N_CLASSES_PER_TASK * self.N_TASKS
        train_loader = DataLoader(TensorDataset(self.train_data, self.train_targets), batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(self.test_data, self.test_targets), batch_size=self.args.val_batch_size, shuffle=False)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        mean = train_loader.dataset.tensors[0].mean((0, 2, 3))
        std = train_loader.dataset.tensors[0].std((0, 2, 3), unbiased=False)
        print(f'Classes: {self.i} - {self.i+self.N_CLASSES_PER_TASK}, mean = {mean}, std = {std}')
        self.i += self.N_CLASSES_PER_TASK
        self.test_transforms += [torch.nn.Sequential(
                K.augmentation.Normalize(mean, std)
            )]
        return train_loader, test_loader


    @staticmethod
    def get_transform():
        return SequentialCIFAR100.train_transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return SequentialCIFAR100.test_transform 

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler

