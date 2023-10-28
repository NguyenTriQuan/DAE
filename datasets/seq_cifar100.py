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
import torch
import random


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class TrainCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.rot = True
        self.non_aug_transform = transforms.ToTensor()
        self.root = root
        super(TrainCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
class TestCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.non_aug_transform = transforms.ToTensor()
        self.num_aug = 0
        self.root = root
        super(TestCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        imgs = []
        if self.transform is not None:
            imgs.append(self.non_aug_transform(img))
            for _ in range(self.num_aug):
                imgs.append(self.transform(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack(imgs, dim=0).squeeze(), target


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = 100
    scale = (0.08, 1.0)
    TRANSFORM = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomResizedCrop(size=(32, 32), scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])

    def get_examples_number(self):
        train_dataset = TrainCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.ToTensor()

        train_dataset = CIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    transform, self.NAME)
        else:
            test_dataset = TestCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

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
    

# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from typing import Tuple

# import torch.nn.functional as F
# import torch.optim
# import torchvision.transforms as transforms
# from backbone.ResNet18 import resnet18
# from PIL import Image
# from torchvision.datasets import CIFAR100

# from datasets.transforms.denormalization import DeNormalize
# from datasets.utils.continual_dataset import (ContinualDataset,
#                                               store_masked_loaders)
# from datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# import kornia as K
# import torch


# class SequentialCIFAR100(ContinualDataset):

#     NAME = 'seq-cifar100'
#     SETTING = 'class-il'
#     N_CLASSES_PER_TASK = 20
#     N_TASKS = 5
#     INPUT_SHAPE = (3, 32, 32)

#     def download(self):
#         self.train_transform = torch.nn.Sequential(
#                     K.augmentation.RandomResizedCrop(size=(32, 32), scale=self.args.scale, p=1, same_on_batch=False),
#                     K.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=False),
#                     K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=False),
#                     # K.augmentation.RandomGrayscale(p=0.2, same_on_batch=False),
#                 )
        
#         self.test_transform = torch.nn.Sequential(
#                     K.augmentation.Normalize((0.5071, 0.4867, 0.4408),
#                                             (0.2675, 0.2565, 0.2761)),
#                 )
#         self.ood_transform = K.augmentation.RandomRotation((90, 270), same_on_batch=False, p=1)
#         self.test_transforms = []
        
#         train_set=CIFAR100(base_path() + 'CIFAR100',train=True,download=True)
#         test_set=CIFAR100(base_path() + 'CIFAR100',train=False,download=True)
#         self.train_data, self.train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
#         self.test_data, self.test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

#         self.train_data = self.train_data.permute(0, 3, 1, 2)/255.0
#         self.test_data = self.test_data.permute(0, 3, 1, 2)/255.0
#         self.N_CLASSES = len(self.train_targets.unique())
#         # print(train_data.mean((0, 2, 3)), train_data.std((0, 2, 3), unbiased=False))


#     def get_data_loaders(self):
#         train_mask = (self.train_targets >= self.i) & (self.train_targets < self.i + self.N_CLASSES_PER_TASK)
#         test_mask = (self.test_targets >= self.i) & (self.test_targets < self.i + self.N_CLASSES_PER_TASK)

#         train_loader = DataLoader(TensorDataset(self.train_data[train_mask].clone(), self.train_targets[train_mask].clone()), batch_size=self.args.batch_size, shuffle=True)
#         test_loader = DataLoader(TensorDataset(self.test_data[test_mask].clone(), self.test_targets[test_mask].clone()), batch_size=self.args.val_batch_size, shuffle=False)
#         self.test_loaders.append(test_loader)
#         self.train_loader = train_loader
#         print(f'Data info: Classes: {self.i}-{self.i+self.N_CLASSES_PER_TASK}, Size: {train_loader.dataset.tensors[0].shape[0]}, Mean: {train_loader.dataset.tensors[0].mean((0,2,3))}, STD: {train_loader.dataset.tensors[0].std((0,2,3))}')
#         self.i += self.N_CLASSES_PER_TASK
#         return train_loader, test_loader
    
#     def get_full_data_loader(self):
#         self.N_CLASSES_PER_TASK = self.N_CLASSES_PER_TASK * self.N_TASKS
#         train_loader = DataLoader(TensorDataset(self.train_data, self.train_targets), batch_size=self.args.batch_size, shuffle=True)
#         test_loader = DataLoader(TensorDataset(self.test_data, self.test_targets), batch_size=self.args.val_batch_size, shuffle=False)
#         self.test_loaders.append(test_loader)
#         self.train_loader = train_loader
#         mean = train_loader.dataset.tensors[0].mean((0, 2, 3))
#         std = train_loader.dataset.tensors[0].std((0, 2, 3), unbiased=False)
#         print(f'Classes: {self.i} - {self.i+self.N_CLASSES_PER_TASK}, mean = {mean}, std = {std}')
#         self.i += self.N_CLASSES_PER_TASK
#         self.test_transforms += [torch.nn.Sequential(
#                 K.augmentation.Normalize(mean, std)
#             )]
#         return train_loader, test_loader


#     @staticmethod
#     def get_transform():
#         return SequentialCIFAR100.train_transform

#     @staticmethod
#     def get_backbone():
#         return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
#                         * SequentialCIFAR100.N_TASKS)

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return SequentialCIFAR100.test_transform 

#     @staticmethod
#     def get_denormalization_transform():
#         transform = DeNormalize((0.5071, 0.4867, 0.4408),
#                                 (0.2675, 0.2565, 0.2761))
#         return transform

#     @staticmethod
#     def get_epochs():
#         return 50

#     @staticmethod
#     def get_batch_size():
#         return 32

#     @staticmethod
#     def get_minibatch_size():
#         return SequentialCIFAR100.get_batch_size()

#     @staticmethod
#     def get_scheduler(model, args) -> torch.optim.lr_scheduler:
#         model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
#         return scheduler

