# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.status import ProgressBar
import torch.nn.functional as F
import wandb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--lamb", type=str, required=False, help="capacity control.", default='0')
    parser.add_argument("--alpha", type=float, required=False, help="maximize entropy of ood samples loss factor.", default=1)
    parser.add_argument("--dropout", type=float, required=False, help="Dropout probability.", default=0.0)
    parser.add_argument("--sparsity", type=float, required=False, help="Super mask sparsity.", default=0)
    parser.add_argument("--temperature", default=0.1, type=float, required=False, help="Supervised Contrastive loss temperature.")
    parser.add_argument("--negative_slope", default=0, type=float, required=False, help="leaky relu activation negative slope.")
    parser.add_argument("--ablation", type=str, required=False, help="Ablation study.", default="")
    parser.add_argument("--norm_type", type=str, required=False, help="batch normalization layer", default="none")
    parser.add_argument("--debug", action="store_true", help="Quick test.")
    parser.add_argument("--verbose", action="store_true", help="compute test accuracy and number of params.")
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--cal", action="store_true", help="calibration training")
    parser.add_argument("--lr_score", type=float, required=False, help="score learning rate.", default=0.1)
    parser.add_argument("--num_tasks", type=int, required=False, help="number of tasks to run.", default=100)
    parser.add_argument("--total_tasks", type=int, required=True, help="total number of tasks.", default=10)
    parser.add_argument("--factor", type=float, required=False, help="entropy scale factor.", default=1)
    parser.add_argument("--num_aug", type=int, required=False, help="number of augument samples used when evaluation.", default=16)
    parser.add_argument("--task", type=int, required=False, help="Specify task for eval or cal.", default=-1)
    parser.add_argument('--buffer_size', type=int, required=False, default=0, help='The size of the memory buffer.')
    return parser

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def ensemble_outputs(outputs):
    # outputs shape [num_member, bs, num_cls]
    outputs = F.log_softmax(outputs, dim=-1)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=0)
    return log_outputs


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, dataset):
        super(Joint, self).__init__(backbone, loss, args, dataset)
        self.net.to(self.device)

    def forward(self, inputs, ba=True):
        bs = inputs.shape[0]
        if ba:
            # batch augmentation
            N = self.args.num_aug
            inputs = inputs.repeat(N, 1, 1, 1)
            x = self.dataset.train_transform(inputs)
        else:
            x = inputs

        outputs = self.net(x)
        if ba:
            outputs = outputs.view(N, bs, -1)
            outputs = ensemble_outputs(outputs)
        
        del x
        return outputs
            

    def evaluate(self, mode="^_^"):
        ba = "ba" in mode

        with torch.no_grad():
            self.net.eval()
            
            cil_correct, total = 0.0, 0.0
            for data in self.dataset.test_loaders[0]:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs, ba)
                cil_correct += torch.sum(outputs.argmax(1) == labels).item()
                total += labels.shape[0]

            acc = round(cil_correct / total * 100, 2)
            return acc

    def train(self, train_loader):
        total = 0
        correct = 0
        total_loss = 0

        self.net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            inputs = self.dataset.train_transform(inputs)
            # inputs = self.dataset.test_transforms[self.task](inputs)
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            assert not math.isnan(loss)
            loss.backward()
            self.opt.step()
            total += bs
            total_loss += loss.item() * bs
            correct += (outputs.argmax(1) == labels).sum().item()
                
        self.scheduler.step()

        return total_loss / total, correct * 100 / total
    
    def train_loop(self):
        n_epochs = 120
        params = self.net.parameters()
        count = 0
        for param in params:
            count += param.numel()
        print(f'Joint Training, Number of optim params: {count}')
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=0, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [100, 115], gamma=0.1, verbose=False)
        
        train_loader, test_loader = self.dataset.get_full_data_loader()
        progress_bar = ProgressBar()
        for epoch in range(n_epochs):
            loss, train_acc = self.train(train_loader)
            if self.args.verbose:
                test_acc = self.evaluate(mode='joint')
                progress_bar.prog(epoch, n_epochs, epoch, 0, loss, train_acc, test_acc)
                wandb.log({"epoch": epoch, "loss": loss, "train acc": train_acc, "test acc": test_acc})

        print()
        mode = 'joint'
        acc = self.evaluate(mode=mode)
        print(f"{mode}: cil {acc}")

        mode = 'joint_ba'
        acc = self.evaluate(mode=mode)
        print(f"{mode}: cil {acc}")
