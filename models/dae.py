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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from itertools import cycle
from backbone.utils.dae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer
import numpy as np

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
    parser.add_argument('--temperature', default=1, type=float, required=False,
                        help='Weighted ensemble temperature.')
    parser.add_argument('--negative_slope', default=0, type=float, required=False,
                        help='leaky relu activation negative slope.')
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
        # self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type='bn_affine_track', args=args)
        self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type=None, args=args)
        self.buffer = None
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
                if 'kbts' in mode:
                    out = self.net(x, i, mode='kbts')
                    outputs.append(out)
                if 'jr' in mode:
                    out = out_jr[:, self.net.DM[-1].shape_out[i]:self.net.DM[-1].shape_out[i+1]]
                    outputs.append(out)
                outputs = ensemble_outputs(outputs)
                outputs_tasks.append(outputs)
                joint_entropy = entropy(outputs.exp())
                joint_entropy_tasks.append(joint_entropy)
            
            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, dim=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            _, predicts = predicted_outputs.max(1)
            return predicts + predicted_task * self.dataset.N_CLASSES_PER_TASK
    
    def train(self, train_loader, progress_bar, mode, squeeze, epoch):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = self.dataset.train_transform(inputs)
            self.opt.zero_grad()
            outputs = self.net(inputs, self.task, mode)
            loss = self.loss(outputs, labels - self.task * self.dataset.N_CLASSES_PER_TASK)
            loss.backward()
            self.opt.step()
            self.net.normalize()
            _, predicts = outputs.max(1)
            correct += torch.sum(predicts == (labels - self.task * self.dataset.N_CLASSES_PER_TASK)).item()
            total += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]
            if squeeze:
                self.net.proximal_gradient_descent(self.args.lr, self.lamb[self.task])
                num_neurons = [m.mask_out.sum().item() for m in self.net.DM[:-1]]
                progress_bar.prog(i, len(train_loader), epoch, self.task, total_loss/total, correct/total*100, 0, num_neurons)
            else:
                progress_bar.prog(i, len(train_loader), epoch, self.task, total_loss/total, correct/total*100)
        if squeeze:
            self.net.squeeze(self.opt.state)
            # self.net.update_strength()

    def train_rehearsal(self, progress_bar, epoch):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0
        if self.buffer is not None:
            buffer_loader = iter(self.buffer)
        for i, logits_data in enumerate(self.logits_loader):
            self.opt.zero_grad()
            logits_data = [tmp.to(self.device) for tmp in logits_data]
            if self.buffer is not None:
                try:
                    buffer_data = next(buffer_loader)
                except StopIteration:
                    buffer_loader = iter(self.buffer)
                    buffer_data = next(buffer_loader)

                buffer_data = [tmp.to(self.device) for tmp in buffer_data]
                for j in range(len(logits_data)):
                    logits_data[j] = torch.cat([buffer_data[j], logits_data[j]])

            outputs = self.net(self.dataset.train_transform(logits_data[0]), self.task, mode='jr')
            loss = self.loss(outputs, logits_data[1])
            for t in range(self.task-1):
                outputs_task = outputs[:, self.net.DM[-1].shape_out[t]:self.net.DM[-1].shape_out[t+1]]
                loss += self.args.alpha * modified_kl_div(smooth(logits_data[t+2], 2, 1), smooth(self.soft(outputs_task), 2, 1))
            loss.backward()
            self.opt.step()
            _, predicts = outputs.max(1)
            correct += torch.sum(predicts == logits_data[1]).item()
            total += logits_data[1].shape[0]
            total_loss += loss.item() * logits_data[1].shape[0]
            progress_bar.prog(i, len(self.logits_loader), epoch, self.task, total_loss/total, correct/total*100)


    def begin_task(self, dataset):
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        if self.task == 0:
            get_related_layers(self.net, self.dataset.INPUT_SHAPE)
        self.net.initialize()
        self.net.check()
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for m in self.net.DM:
            m.kbts_sparsities.append(m.sparsity)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=0, momentum=self.args.optim_mom)

    def end_task(self, dataset) -> None:
        self.net.set_jr_params()
        self.task += 1
        self.net.freeze()
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for m in self.net.DM:
            m.jr_sparsity = m.sparsity

    def get_rehearsal_logits(self, train_loader):
        self.net.eval()
        data = [[] for _ in range(self.task+2)]
        for inputs, labels in train_loader:
            data[0].append(inputs)
            data[1].append(labels)
            inputs = inputs.to(self.device)
            inputs = self.dataset.test_transform(inputs)
            for i in range(self.task):
                outputs = [self.net(inputs, i, mode='ets'), self.net(inputs, i, mode='kbts')]
                data[i+2].append(ensemble_outputs(outputs).exp().detach().clone().cpu())

        data = [torch.cat(temp) for temp in data]
        self.logits_loader = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)

        if self.buffer is not None:
            new_logits = []
            for buffer_data in self.buffer:
                inputs = buffer_data[0].to(self.device)
                inputs = self.dataset.test_transform(inputs)
                outputs = [self.net(inputs, self.task-1, mode='ets'), self.net(inputs, self.task-1, mode='kbts')]
                new_logits.append(ensemble_outputs(outputs).exp().detach().clone().cpu())

            new_logits = torch.cat(new_logits)
            data = list(self.buffer.dataset.tensors)
            data.append(new_logits)
            self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)


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
        samples_per_class = self.args.buffer_size // (self.dataset.N_CLASSES_PER_TASK * self.task)
        
        data = [[] for _ in range(self.task+2)]
        if self.task > 1:
            for _y in self.buffer.dataset.tensors[1].unique():
                idx = (self.buffer.dataset.tensors[1] == _y)
                for i in range(len(self.buffer.dataset.tensors)):
                    data[i] += [self.buffer.dataset.tensors[i][idx][:samples_per_class]]
                # inputs = self.dataset.test_transform(data[0].to(self.device))
                # outputs = [self.net(inputs, self.task-1, mode='ets'), self.net(inputs, self.task-1, mode='kbts')]
                # data[-1].append(ensemble_outputs(outputs).exp().detach().clone().cpu())

        classes_start, classes_end = (self.task-1) * self.dataset.N_CLASSES_PER_TASK, self.task * self.dataset.N_CLASSES_PER_TASK
        print(f'Filling Buffer: samples per class {samples_per_class}, classes start {classes_start}, classes end {classes_end}')

        for _y in self.logits_loader.dataset.tensors[1].unique():
            idx = (self.logits_loader.dataset.tensors[1] == _y)
            ents = entropy(self.logits_loader.dataset.tensors[-1])
            #select samples with highest entropy
            values, indices = ents[idx].sort(dim=0, descending=True)
            for i in range(len(self.logits_loader.dataset.tensors)):
                data[i] += [self.logits_loader.dataset.tensors[i][idx][indices[:samples_per_class]]]
        
        data = [torch.cat(temp) for temp in data]
        self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        print('Buffer size:', data[0].shape)
        self.net.train(mode)


def get_related_layers(net, input_shape):
    def forward_hook(m, i, o):
        m.output_idx = torch.rand(1).cuda() * m.input_idx
        o[0] += m.output_idx
        return
    
    def forward_pre_hook(m, i):
        m.input_idx = i[0].mean().detach()
        return (torch.zeros_like(i[0]), i[1], i[2])

    handes = []
    for m in net.DM:
        m.input_idx = torch.tensor(-np.inf).cuda()
        m.output_idx = torch.tensor(-np.inf).cuda()
        h1 = m.register_forward_pre_hook(forward_pre_hook)
        h2 = m.register_forward_hook(forward_hook)
        handes += [h1, h2]
    c, h, w = input_shape
    data = torch.ones((1, c, h, w), dtype=torch.float, device='cuda')
    net.eval()
    out = net(data, 0, mode='ets')  
    for h in handes:
        h.remove()
    
    for n, m in net.named_modules():
        if isinstance(m, _DynamicLayer):
            # print(n, m.base_in_features, m.base_out_features, m.input_idx, m.output_idx)
            m.name = n
            m.prev_layers = []
            m.next_layers = []
    eps = 1e-6
    for i in range(len(net.DM)):
        for j in range(i):
            if (net.DM[i].input_idx - net.DM[j].output_idx).abs() < eps:
                net.DM[i].prev_layers.append(net.DM[j])
                net.DM[j].next_layers.append(net.DM[i])
            else:
                for k in range(j):
                    if (net.DM[i].input_idx - net.DM[j].output_idx - net.DM[k].output_idx).abs() < eps:
                        net.DM[i].prev_layers += [net.DM[j], net.DM[k]]
                        net.DM[j].next_layers += [net.DM[i]]
                        net.DM[k].next_layers += [net.DM[i]]

                    if (net.DM[i].input_idx - net.DM[j].output_idx - net.DM[k].input_idx).abs() < eps:
                        net.DM[i].prev_layers += [net.DM[j]] + net.DM[k].prev_layers
                        net.DM[j].next_layers += [net.DM[i]]
                        for g in net.DM[k].prev_layers:
                            g.next_layers += [net.DM[i]]

                    if (net.DM[i].input_idx - net.DM[k].output_idx - net.DM[j].input_idx).abs() < eps:
                        net.DM[i].prev_layers += [net.DM[k]] + net.DM[j].prev_layers
                        net.DM[k].next_layers += [net.DM[i]]
                        for g in net.DM[j].prev_layers:
                            g.next_layers += [net.DM[i]]

                    if (net.DM[i].input_idx - net.DM[j].input_idx - net.DM[k].input_idx).abs() < eps:
                        net.DM[i].prev_layers += net.DM[j].prev_layers + net.DM[k].prev_layers
                        for g in net.DM[j].prev_layers:
                            g.next_layers += [net.DM[i]]
                        for g in net.DM[k].prev_layers:
                            g.next_layers += [net.DM[i]]

    net.prev_layers = []
    for m in net.DM:
        m.prev_layers = tuple(set(m.prev_layers))
        m.next_layers = tuple(set(m.next_layers))
        if len(m.prev_layers) != 0:
            net.prev_layers.append(m.prev_layers)
        # print(m.name, m.base_in_features, m.base_out_features, m.input_idx, m.output_idx)
        # for j, n in enumerate(m.prev_layers):
        #     print('prev', j, n.name, n.base_in_features, n.base_out_features, n.input_idx, n.output_idx)
        # for j, n in enumerate(m.next_layers):
        #     print('next', j, n.name, n.base_in_features, n.base_out_features, n.input_idx, n.output_idx)
        # print()
    net.prev_layers = tuple(set(net.prev_layers))
    for i, layers in enumerate(net.prev_layers):
        if len(layers) == 0:
            print(i, layers)
        for m in layers:
            print(i, m.name, m.base_in_features, m.base_out_features)

