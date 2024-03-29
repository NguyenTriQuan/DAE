# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
import torch
import random

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.non_aug_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        self.rot_transform = transforms.RandomRotation(degrees=(90, 270))
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download

                print('Downloading dataset')
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
                download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class TrainTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(TrainTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)
        self.rot = True

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class TestTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(TestTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)
        self.num_aug = 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        imgs = []
        if self.transform is not None:
            if self.num_aug == 0:
                imgs.append(self.non_aug_transform(img))
            else:
                for _ in range(self.num_aug):
                    imgs.append(self.transform(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack(imgs, dim=0).squeeze(), target


class SequentialTinyImagenet(ContinualDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    N_CLASSES = 200
    scale = (0.08, 1.0)
    TRANSFORM = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomResizedCrop(size=(32, 32), scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()])

        train_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                       train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = TestTinyImagenet(base_path() + 'TINYIMG',
                                        train=False, download=True, transform=transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone():
        return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
                        * SequentialTinyImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                (0.2770, 0.2691, 0.2821))
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
        return SequentialTinyImagenet.get_batch_size()
    

# # Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# # All rights reserved.
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.

# import os
# from typing import Optional

# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from backbone.ResNet18 import resnet18
# from PIL import Image
# from torch.utils.data import Dataset

# from datasets.transforms.denormalization import DeNormalize
# from datasets.utils.continual_dataset import (ContinualDataset,
#                                               store_masked_loaders)
# from datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path
# import torch
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# import kornia as K

# class TinyImagenet(Dataset):
#     """
#     Defines Tiny Imagenet as for the others pytorch datasets.
#     """

#     def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
#                  target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
#         self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
#         self.root = root
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.download = download

#         if download:
#             if os.path.isdir(root) and len(os.listdir(root)) > 0:
#                 print('Download not needed, files already on disk.')
#             else:
#                 from onedrivedownloader import download

#                 print('Downloading dataset')
#                 ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
#                 download(ln, filename=os.path.join(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

#         resize = K.augmentation.Resize(size=(32, 32))
#         self.data = []
#         for num in range(20):
#             sub_data = np.load(os.path.join(
#                 root, 'processed/x_%s_%02d.npy' %
#                       ('train' if self.train else 'val', num + 1)))
#             sub_data = torch.FloatTensor(sub_data)
#             sub_data = sub_data.permute(0, 3, 1, 2)
#             sub_data = resize(sub_data)
#             self.data.append(sub_data)
#         self.data = torch.cat(self.data, dim=0)

#         self.targets = []
#         for num in range(20):
#             sub_targets = np.load(os.path.join(
#                 root, 'processed/y_%s_%02d.npy' %
#                       ('train' if self.train else 'val', num + 1)))
#             sub_targets = torch.LongTensor(sub_targets)
#             self.targets.append(sub_targets)
#         self.targets = torch.cat(self.targets, dim=0)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(np.uint8(255 * img))
#         original_img = img.copy()

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         if hasattr(self, 'logits'):
#             return img, target, original_img, self.logits[index]

#         return img, target


# class MyTinyImagenet(TinyImagenet):
#     """
#     Defines Tiny Imagenet as for the others pytorch datasets.
#     """

#     def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
#                  target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
#         super(MyTinyImagenet, self).__init__(
#             root, train, transform, target_transform, download)

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(np.uint8(255 * img))
#         original_img = img.copy()

#         not_aug_img = self.not_aug_transform(original_img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         if hasattr(self, 'logits'):
#             return img, target, not_aug_img, self.logits[index]

#         return img, target, not_aug_img


# class SequentialTinyImagenet(ContinualDataset):

#     NAME = 'seq-tinyimg'
#     SETTING = 'class-il'
#     N_CLASSES_PER_TASK = 20
#     N_TASKS = 10
#     INPUT_SHAPE = (3, 32, 32)

#     def download(self):
#         self.train_transform = torch.nn.Sequential(
#                     K.augmentation.RandomResizedCrop(size=(32, 32), scale=self.args.scale, p=1, same_on_batch=False),
#                     K.augmentation.RandomHorizontalFlip(p=0.5, same_on_batch=False),
#                     K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8, same_on_batch=False),
#                     # K.augmentation.RandomGrayscale(p=0.2, same_on_batch=False),
#                 )
#         self.ood_transform = K.augmentation.RandomRotation((90, 270), same_on_batch=False, p=1)
#         train_set = TinyImagenet(base_path() + 'TINYIMG', train=True, download=True)
#         test_set = TinyImagenet(base_path() + 'TINYIMG', train=False, download=True)
#         self.train_data, self.train_targets = torch.FloatTensor(train_set.data), torch.LongTensor(train_set.targets)
#         self.test_data, self.test_targets = torch.FloatTensor(test_set.data), torch.LongTensor(test_set.targets)
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
#     def get_backbone():
#         return resnet18(SequentialTinyImagenet.N_CLASSES_PER_TASK
#                         * SequentialTinyImagenet.N_TASKS)

#     @staticmethod
#     def get_loss():
#         return F.cross_entropy

#     def get_transform(self):
#         transform = transforms.Compose(
#             [transforms.ToPILImage(), self.TRANSFORM])
#         return transform

#     @staticmethod
#     def get_normalization_transform():
#         transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
#                                          (0.2770, 0.2691, 0.2821))
#         return transform

#     @staticmethod
#     def get_denormalization_transform():
#         transform = DeNormalize((0.4802, 0.4480, 0.3975),
#                                 (0.2770, 0.2691, 0.2821))
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
#         return SequentialTinyImagenet.get_batch_size()
