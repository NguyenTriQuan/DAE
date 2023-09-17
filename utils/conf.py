# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np
import os

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    # os.makedirs('./tmp/log/', exist_ok=True)
    path = '/cm/archive/quannt40/dae/log/'
    os.makedirs(path, exist_ok=True)
    return path

def base_path_dataset() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    path = '/cm/archive/quannt40/dae/data/'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + 'CIFAR10', exist_ok=True)
    os.makedirs(path + 'CIFAR100', exist_ok=True)
    os.makedirs(path + 'TINYIMG', exist_ok=True)
    # os.makedirs('./tmp/data/', exist_ok=True)
    return path

def base_path_memory() -> str:
    """
    Returns the base bath where to store model and buffer.
    """
    # os.makedirs('./tmp/memory/', exist_ok=True)
    path = '/cm/archive/quannt40/dae/memory/'
    os.makedirs(path, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
