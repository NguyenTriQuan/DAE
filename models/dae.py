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
from backbone.ResNet18_DAE import resnet18

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
    parser.add_argument('--fix', action='store_true',
                        help='Do not expand the network.')
    parser.add_argument('--debug', action='store_true',
                        help='Quick test.')
    return parser

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def ensemble_outputs(outputs):
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    outputs = torch.stack(outputs, dim=-1)
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=-1)
    return log_outputs

class DAE(ContinualModel):
    NAME = 'DAE'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(DAE, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)
        self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type='bn', args=args)
        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

    def forward(self, x, t=None, ets=False, kbts=False, jr=False):
        if t is not None:
            self.net.get_kb_params(t)
            outputs = []
            if ets:
                outputs.append(self.net(x, t, mode='ets'))
            if kbts:
                outputs.append(self.net(x, t, mode='kbts'))
            if jr:
                self.net.get_kb_params(self.task)
                outputs.append(self.net(x, self.task, mode='jr'))
            outputs = ensemble_outputs(outputs)
            _, predicts = outputs.max(1)
            return predicts + t * self.dataset.N_CLASSES_PER_TASK
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            for i in range(self.task+1):
                self.net.get_kb_params(i)
                outputs = []
                if ets:
                    outputs.append(self.net(x, i, mode='ets'))
                if kbts:
                    outputs.append(self.net(x, i, mode='kbts'))
                if jr:
                    self.net.get_kb_params(self.task)
                    outputs.append(self.net(x, self.task, mode='jr'))
                outputs = ensemble_outputs(outputs)
                outputs_tasks.append(outputs)
                joint_entropy = -torch.sum(outputs * torch.log(outputs+0.0001), dim=1)
                joint_entropy_tasks.append(joint_entropy)
            
            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, axis=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            _, predicts = predicted_outputs.max(1)
            return predicts + predicted_task * self.dataset.N_CLASSES_PER_TASK

    def observe(self, inputs, labels, not_aug_inputs, mode):

        self.opt.zero_grad()

        if not self.buffer.is_empty() and mode == 'jr':
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            outputs = self.net(buf_inputs, self.task, mode)
            loss = self.loss(outputs, buf_labels)
        else:
            outputs = self.net(inputs, self.task, mode)
            loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()


    def begin_task(self, dataset):
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        self.opt = torch.optim.SGD(self.net.get_optim_params(), lr=self.args.lr)
        self.net.get_kb_params(self.task)

    def end_task(self, dataset) -> None:
        self.net.get_jr_params()
        self.task += 1
        self.net.freeze()
        self.net.clear_memory()
        self.net.get_kb_params(self.task)
        print(self.task)