import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar
from utils.conf import base_path_memory
from utils.lars_optimizer import LARC
# try:
#     import wandb
# except ImportError:
#     wandb = None

import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6,7"

def train(model, args, train_loader, mode, checkpoint=None, t=0):
    
    progress_bar = ProgressBar()
    for epoch in range(start_epoch, n_epochs):
        if cal:
            loss, train_acc = model.back_updating(train_loader, t)
        else:          
            loss, train_acc = model.train_contrast(train_loader, mode, ets, kbts, rot, buf, adv, feat, squeeze, augment)

        # wandb.save(base_path_memory() + args.title + '.tar')
        if args.verbose:
            test_acc = 0
            if not feat and not cal:
                test_acc = model.evaluate(task=model.task, mode=mode)
            if squeeze:
                num_params, num_neurons = model.net.count_params()
                num_neurons = '-'.join(str(int(num)) for num in num_neurons)
                num_params = sum(num_params)
                progress_bar.prog(epoch, n_epochs, epoch, model.task, loss, train_acc, test_acc, num_params, num_neurons)
                wandb.log({'epoch':epoch, f"Task {model.task} {mode} loss": loss, f"Task {model.task} {mode} train acc": train_acc,
                           f"Task {model.task} {mode} test acc": test_acc, f"Task {model.task} params": num_params})
            else:
                progress_bar.prog(epoch, n_epochs, epoch, model.task, loss, train_acc, test_acc)
                wandb.log({'epoch':epoch, f"Task {model.task} {mode} loss": loss, f"Task {model.task} {mode} train acc": train_acc,
                           f"Task {model.task} {mode} test acc": test_acc})

        #save model
        model.net.clear_memory()
        checkpoint = {'net': model.net, 'opt': model.opt, 'scheduler': model.scheduler, 'epoch':epoch, 'mode':mode, 'task':model.task}
        torch.save(checkpoint, base_path_memory() + args.title + '.checkpoint')

        if epoch >= num_squeeze:
            squeeze = False

    print()
    if ets:
        num_params, num_neurons = model.net.count_params()
        num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
