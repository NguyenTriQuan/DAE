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
        self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type='bn_affine_track', args=args)
        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

    def forward(self, x, t=None, mode='ets_kbts_jr'):
        if t is not None:
            self.net.get_kb_params(t)
            outputs = []
            if 'ets' in mode:
                outputs.append(self.net(x, t, mode='ets'))
            if 'kbts' in mode:
                outputs.append(self.net(x, t, mode='kbts'))
            if 'jr' in mode:
                self.net.get_kb_params(self.task)
                out_jr = self.net(x, self.task, mode='jr')
                out_jr = out_jr[:, self.net.DM[-1].shape_out[t]:self.net.DM[-1].shape_out[t+1]]
                outputs.append(out_jr)
            outputs = ensemble_outputs(outputs)
            _, predicts = outputs.max(1)
            return predicts + t * self.dataset.N_CLASSES_PER_TASK
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            if 'jr' in mode:
                self.net.get_kb_params(self.task)
                out_jr = self.net(x, self.task, mode='jr')
            for i in range(self.task):
                self.net.get_kb_params(i)
                outputs = []
                if 'ets' in mode:
                    outputs.append(self.net(x, i, mode='ets'))
                if 'kbts' in mode:
                    outputs.append(self.net(x, i, mode='kbts'))
                if 'jr' in mode:
                    outputs.append(out_jr[:, self.net.DM[-1].shape_out[i]:self.net.DM[-1].shape_out[i+1]])
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

        if not self.buffer.is_empty() and 'jr' in mode:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            outputs = self.net(buf_inputs, self.task, mode)
            loss = self.loss(outputs, buf_labels)
        else:
            outputs = self.net(inputs, self.task, mode)
            loss = self.loss(outputs, labels - self.task * self.dataset.N_CLASSES_PER_TASK)

        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()


    def begin_task(self, dataset):
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        # self.opt = torch.optim.SGD(self.net.get_optim_params(), lr=self.args.lr)
        self.net.get_kb_params(self.task)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        # for n, p in self.net.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)

    def end_task(self, dataset) -> None:
        self.net.get_jr_params()
        self.task += 1
        self.net.freeze()
        # self.net.clear_memory()
        self.net.get_kb_params(self.task)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)

    def fill_buffer(self) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()
        samples_per_class = self.buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS)

        # if t_idx > 0:
        #     # 1) First, subsample prior classes
        #     buf_x, buf_y, buf_l = self.buffer.get_all_data()

        #     mem_buffer.empty()
        #     for _y in buf_y.unique():
        #         idx = (buf_y == _y)
        #         _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
        #         mem_buffer.add_data(
        #             examples=_y_x[:samples_per_class],
        #             labels=_y_y[:samples_per_class],
        #             logits=_y_l[:samples_per_class]
        #         )

        # 2) Then, fill with current tasks
        loader = self.dataset.train_loader
        norm_trans = self.dataset.get_normalization_transform()
        if norm_trans is None:
            def norm_trans(x): return x
        classes_start, classes_end = self.task * self.dataset.N_CLASSES_PER_TASK, (self.task + 1) * self.dataset.N_CLASSES_PER_TASK

        # 2.1 Extract all features
        a_x, a_y, a_l = [], [], []
        for x, y, not_norm_x in loader:
            mask = (y >= classes_start) & (y < classes_end)
            x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
            if not x.size(0):
                continue
            x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
            outs = self.net(norm_trans(not_norm_x), self.task, mode='ets')
            a_l.append(self.loss(outs, y - self.task * self.dataset.N_CLASSES_PER_TASK))
        a_x, a_y, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_l)

        # 2.2 Compute class means
        # for _y in a_y.unique():
        #     idx = (a_y == _y)
        #     _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
        #     feats = a_f[idx]
        #     mean_feat = feats.mean(0, keepdim=True)

        #     running_sum = torch.zeros_like(mean_feat)
        #     i = 0
        #     while i < samples_per_class and i < feats.shape[0]:
        #         cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

        #         idx_min = cost.argmin().item()

        #         mem_buffer.add_data(
        #             examples=_x[idx_min:idx_min + 1].to(self.device),
        #             labels=_y[idx_min:idx_min + 1].to(self.device),
        #             logits=_l[idx_min:idx_min + 1].to(self.device)
        #         )

        #         running_sum += feats[idx_min:idx_min + 1]
        #         feats[idx_min] = feats[idx_min] + 1e6
        #         i += 1

        # assert len(mem_buffer.examples) <= mem_buffer.buffer_size
        # assert mem_buffer.num_seen_examples <= mem_buffer.buffer_size

        # self.net.train(mode)