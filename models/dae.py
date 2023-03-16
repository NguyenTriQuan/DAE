# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, icarl_replay


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Dynamic Architecture and Ensemble of Knowledge Base.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--lamb', type=float, required=True,
                        help='capacity control.')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout probability.')
    parser.add_argument('--sparsity', type=float, required=True,
                        help='Super mask sparsity.')
    parser.add_argument('--ablation', type=str, required=False,
                        help='Ablation study.', default='')
    return parser


class DAE(ContinualModel):
    NAME = 'DAE'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(DAE, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

    def forward(self, x, t, kbts, jr):
        return self.net(x, t, kbts, jr)

    def observe(self, inputs, labels, not_aug_inputs, t, kbts=False, jr=False):

        self.opt.zero_grad()

        if not self.buffer.is_empty() and jr:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            outputs = self.net(buf_inputs, t, kbts, jr)
            loss = self.loss(outputs, buf_labels)
        else:
            outputs = self.net(inputs, t, kbts, jr)
            loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()


    def begin_task(self, dataset):
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        if self.task > 0:
            self.net.freeze()
        self.opt = torch.optim.SGD(self.net.get_optim_params(), lr=self.args.lr)

    def end_task(self, dataset) -> None:
        self.task += 1
        self.net.clear_memory()