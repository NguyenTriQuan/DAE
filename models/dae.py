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
    parser.add_argument('--lamb', type=str, required=True,
                        help='capacity control.')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Join Rehearsal Distillation penalty weight.')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout probability.')
    parser.add_argument('--sparsity', type=float, required=True,
                        help='Super mask sparsity.')
    parser.add_argument('--temperature', type=float, required=True,
                        help='Weighted ensemble temperature.')
    parser.add_argument('--ablation', type=str, required=False,
                        help='Ablation study.', default='')
    parser.add_argument('--debug', action='store_true',
                        help='Quick test.')
    return parser

def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)

def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))

def entropy(x):
    return -torch.sum(x * torch.log(x), dim=1)

def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def ensemble_outputs(outputs):
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    outputs = torch.stack(outputs, dim=-1) #[bs, num_cls, num_member]
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=-1)
    return log_outputs

def weighted_ensemble(outputs, weights, temperature):
    outputs = torch.stack(outputs, dim=-1) #[bs, num_cls, num_member]
    weights = torch.stack(weights, dim=-1) #[bs, num_member]

    weights = F.softmax(weights / temperature, dim=-1).unsqueeze(1) #[bs, 1, num_member]
    outputs = F.log_softmax(outputs, dim=-2)
    output_max, _ = torch.max(outputs, dim=-1, keepdim=True)
    log_outputs = output_max + torch.log(torch.sum((outputs - output_max).exp() * weights, dim=-1, keepdim=True))
    return log_outputs.squeeze(-1)

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
        self.lamb = [float(i) for i in args.lamb.split('_')]
        if len(self.lamb) < self.dataset.N_TASKS:
            self.lamb = [self.lamb[-1] if i>=len(self.lamb) else self.lamb[i] for i in range(self.dataset.N_TASKS)]
        print('lambda tasks', self.lamb)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x, t=None, mode='ets_kbts_jr'):
        if t is not None:
            outputs = []
            if 'jr' in mode:
                out_jr = self.net(x, self.task, mode='jr')
                out_jr = out_jr[:, self.net.DM[-1].shape_out[t]:self.net.DM[-1].shape_out[t+1]]
                outputs.append(out_jr)
            if 'ets' in mode:
                outputs.append(self.net(x, t, mode='ets'))
            if 'kbts' in mode:
                outputs.append(self.net(x, t, mode='kbts'))
            outputs = ensemble_outputs(outputs)
            _, predicts = outputs.max(1)
            return predicts + t * self.dataset.N_CLASSES_PER_TASK
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            if 'jr' in mode:
                out_jr = self.net(x, self.task, mode='jr')
            for i in range(self.task):
                outputs = []
                weights = []
                if 'ets' in mode:
                    out = self.net(x, i, mode='ets')
                    outputs.append(out)
                    weights.append(-entropy(F.softmax(out, dim=1)))
                if 'kbts' in mode:
                    out = self.net(x, i, mode='kbts')
                    outputs.append(out)
                    weights.append(-entropy(F.softmax(out, dim=1)))
                if 'jr' in mode:
                    out = out_jr[:, self.net.DM[-1].shape_out[i]:self.net.DM[-1].shape_out[i+1]]
                    outputs.append(out)
                    weights.append(-entropy(F.softmax(out, dim=1)))
                outputs = ensemble_outputs(outputs)
                # outputs = weighted_ensemble(outputs, weights, self.args.temperature)
                outputs_tasks.append(outputs)
                outputs = outputs.exp()
                # outputs = outputs / outputs.sum(1).view(-1, 1)
                # print(outputs.sum(1))
                joint_entropy = entropy(outputs)
                # p =self.dataset.N_CLASSES_PER_TASK // self.dataset.N_CLASSES_PER_TASK
                # joint_entropy /= p
                joint_entropy_tasks.append(joint_entropy)
            
            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, axis=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            _, predicts = predicted_outputs.max(1)
            return predicts + predicted_task * self.dataset.N_CLASSES_PER_TASK

    def observe(self, inputs, labels, not_aug_inputs, mode):

        self.opt.zero_grad()

        if 'jr' in mode:
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                inputs = torch.cat([inputs, buf_inputs], dim=0)
                labels = torch.cat([labels, buf_labels], dim=0)
            
            outputs = self.net(inputs, self.task, mode)
            loss = self.loss(outputs, labels)
            # distillation loss
            for i in range(self.task):
                out_task = self.net(inputs, i, mode='ets')
                logits = outputs[:, self.net.DM[-1].shape_out[i]:self.net.DM[-1].shape_out[i+1]]

                loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits), 2, 1),
                                                    smooth(self.soft(out_task), 2, 1))
        else:
            outputs = self.net(inputs, self.task, mode)
            loss = self.loss(outputs, labels - self.task * self.dataset.N_CLASSES_PER_TASK)

        loss.backward()
        self.opt.step()

        return loss.item()


    def begin_task(self, dataset):
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)

    def end_task(self, dataset) -> None:
        self.net.set_jr_params()
        self.task += 1
        self.net.freeze()
        self.net.update_scale()
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for n, p in self.net.named_parameters():
            print(n, p.requires_grad)

    def fill_buffer(self, train_loader) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()
        samples_per_class = self.buffer.buffer_size // (self.dataset.N_CLASSES_PER_TASK * self.task)

        if self.task > 1:
        # 1) First, subsample prior classes
            buf_x, buf_y = self.buffer.get_all_data()

            self.buffer.empty()
            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]
                self.buffer.add_data(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class],
                )

        loader = train_loader
        norm_trans = self.dataset.get_normalization_transform()
        if norm_trans is None:
            def norm_trans(x): return x
        classes_start, classes_end = (self.task-1) * self.dataset.N_CLASSES_PER_TASK, self.task * self.dataset.N_CLASSES_PER_TASK

        a_x, a_y, a_l = [], [], []
        for x, y, not_norm_x in loader:
            mask = (y >= classes_start) & (y < classes_end)
            x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
            if not x.size(0):
                continue
            x, y, not_norm_x = (a.to(self.device) for a in (x, y, not_norm_x))
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
            outs_ets = self.net(norm_trans(not_norm_x), self.task-1, mode='ets')
            outs_kbts = self.net(norm_trans(not_norm_x), self.task-1, mode='kbts')
            loss_ets = self.loss(outs_ets, y - (self.task-1) * self.dataset.N_CLASSES_PER_TASK, reduce=False).cpu()
            loss_kbts = self.loss(outs_kbts, y - (self.task-1) * self.dataset.N_CLASSES_PER_TASK, reduce=False).cpu()
            a_l.append(loss_ets+loss_kbts)
        a_x, a_y, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_l)
        print(samples_per_class, classes_start, classes_end, a_x.shape, a_y.shape, a_l.shape)

        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
            values, indices = _l.sort(dim=0, descending=True)
            # print(values.shape, values)
            #select samples with highest loss
            self.buffer.add_data(_x[indices[:samples_per_class]], _y[indices[:samples_per_class]])

        assert len(self.buffer.examples) <= self.buffer.buffer_size
        assert self.buffer.num_seen_examples <= self.buffer.buffer_size

        self.net.train(mode)