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

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
import torch
import random

class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class TrainCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.rot = True
        self.non_aug_transform = transforms.ToTensor()
        self.root = root
        super(TrainCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

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

        if self.rot:
            rot = random.randint(1, 3)
            return img, torch.rot90(img, rot, dims=(1, 2)), target
        else:
            return img, target
    
class TestCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.num_aug = 0
        self.non_aug_transform = transforms.ToTensor()
        self.root = root
        super(TestCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int):
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


class SequentialCIFAR10(ContinualDataset):

    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = 10
    TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(size=(32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.ToTensor()

        train_dataset = TrainCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    transform, self.NAME)
        else:
            test_dataset = TestCIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

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
    

# # Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# # All rights reserved.
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# from typing import Tuple

# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from backbone.ResNet18 import resnet18
# from PIL import Image
# from torchvision.datasets import CIFAR10

# from utils.conf import base_path_dataset as base_path
# from datasets.transforms.denormalization import DeNormalize
# from datasets.utils.continual_dataset import (ContinualDataset,
#                                               store_masked_loaders)
# from datasets.utils.validation import get_train_val
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# import kornia as K
# import torch

# class SequentialCIFAR10(ContinualDataset):

#     NAME = 'seq-cifar10'
#     SETTING = 'class-il'
#     N_CLASSES_PER_TASK = 2
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
#                     K.augmentation.Normalize((0.4914, 0.4822, 0.4465),
#                                             (0.2470, 0.2435, 0.2615)),
#                 )
#         self.ood_transform = K.augmentation.RandomRotation((90, 270), same_on_batch=False, p=1)
#         train_set=CIFAR10(base_path() + 'CIFAR10',train=True,download=True)
#         test_set=CIFAR10(base_path() + 'CIFAR10',train=False,download=True)
#         self.train_data, self.train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
#         self.test_data, self.test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)

#         self.train_data = self.train_data.permute(0, 3, 1, 2)/255.0
#         self.test_data = self.test_data.permute(0, 3, 1, 2)/255.0
#         self.N_CLASSES = len(self.train_targets.unique())

#     def get_data_loaders(self):
#         train_mask = (self.train_targets >= self.i) & (self.train_targets < self.i + self.N_CLASSES_PER_TASK)
#         test_mask = (self.test_targets >= self.i) & (self.test_targets < self.i + self.N_CLASSES_PER_TASK)

#         train_loader = DataLoader(TensorDataset(self.train_data[train_mask], self.train_targets[train_mask]), batch_size=self.args.batch_size, shuffle=True)
#         test_loader = DataLoader(TensorDataset(self.test_data[test_mask], self.test_targets[test_mask]), batch_size=self.args.val_batch_size, shuffle=False)
#         self.test_loaders.append(test_loader)
#         self.train_loader = train_loader
#         print(f'Data info: Classes: {self.i}-{self.i+self.N_CLASSES_PER_TASK}, Size: {train_loader.dataset.tensors[0].shape[0]}, Mean: {train_loader.dataset.tensors[0].mean((0,2,3))}, STD: {train_loader.dataset.tensors[0].std((0,2,3))}')
        
#         self.i += self.N_CLASSES_PER_TASK
#         return train_loader, test_loader

#     @staticmethod
#     def get_transform():
#         return SequentialCIFAR10.train_transform

#     @staticmethod
#     def get_backbone():
#         return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
#                         * SequentialCIFAR10.N_TASKS)

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     @staticmethod
#     def get_normalization_transform():
#         return SequentialCIFAR10.test_transform 

#     @staticmethod
#     def get_denormalization_transform():
#         transform = DeNormalize((0.4914, 0.4822, 0.4465),
#                                 (0.2470, 0.2435, 0.2615))
#         return transform

#     @staticmethod
#     def get_scheduler(model, args):
#         return None

#     @staticmethod
#     def get_epochs():
#         return 50

#     @staticmethod
#     def get_batch_size():
#         return 32

#     @staticmethod
#     def get_minibatch_size():
#         return SequentialCIFAR10.get_batch_size()
