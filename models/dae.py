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
from backbone.ResNet18_DAE import resnet18, resnet10
from torch.utils.data import DataLoader, Dataset, TensorDataset
from itertools import cycle
from backbone.utils.dae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer, DynamicNorm
import numpy as np
import random
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Dynamic Architecture and Ensemble of Knowledge Base.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--lamb', type=str, required=True,
                        help='capacity control.')
    parser.add_argument('--alpha', type=float, required=False,
                        help='Join Rehearsal Distillation penalty weight.', default=0)
    parser.add_argument('--dropout', type=float, required=False,
                        help='Dropout probability.', default=0.0)
    parser.add_argument('--sparsity', type=float, required=True,
                        help='Super mask sparsity.')
    parser.add_argument('--temperature', default=0.3, type=float, required=False,
                        help='Supervised Contrastive loss temperature.')
    parser.add_argument('--negative_slope', default=0, type=float, required=False,
                        help='leaky relu activation negative slope.')
    parser.add_argument('--ablation', type=str, required=False,
                        help='Ablation study.', default='')
    parser.add_argument('--norm_type', type=str, required=False,
                        help='batch normalization layer', default='none')
    parser.add_argument('--debug', action='store_true',
                        help='Quick test.')
    parser.add_argument('--verbose', action='store_true',
                        help='compute test accuracy and number of params.')
    parser.add_argument('--eval', action='store_true',
                        help='evaluation only')
    parser.add_argument('--cal', action='store_true',
                        help='calibration training')
    parser.add_argument('--lr_score', type=float, required=False,
                        help='score learning rate.', default=0.1)
    parser.add_argument('--num_tasks', type=int, required=False,
                        help='number of tasks to run.', default=100)
    parser.add_argument('--total_tasks', type=int, required=True,
                        help='total number of tasks.', default=10)
    parser.add_argument('--factor', type=float, required=False,
                        help='entropy scale factor.', default=1)
    parser.add_argument('--num_aug', type=int, required=False,
                        help='number of augument samples used when evaluation.', default=16)
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

def sup_con_loss(features, labels, temperature):
    features = F.normalize(features, dim=1)
    sim = torch.div(
        torch.matmul(features, features.T),
        temperature)
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(pos_mask),
        1,
        torch.arange(features.shape[0]).view(-1, 1).to(device),
        0
    )
    pos_mask = pos_mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.mean(1, keepdim=True))

    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)        
    # loss
    loss = - mean_log_prob_pos
    loss = loss.mean()

    return loss

class DAE(ContinualModel):
    NAME = 'DAE'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(DAE, self).__init__(backbone, loss, args, transform)
        if args.norm_type == 'none':
            norm_type = None
        else:
            norm_type = args.norm_type

        if args.debug:
            self.net = resnet10(0, norm_type=norm_type, args=args)
            # self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type=norm_type, args=args)
        else:
            self.net = resnet18(0, norm_type=norm_type, args=args)
        self.task = -1
        try:
            self.lamb = float(args.lamb)
        except:
            self.lamb = [float(i) for i in args.lamb.split('_')][0]
        # if len(self.lamb) < self.args.total_tasks:
        #     self.lamb = [self.lamb[-1] if i>=len(self.lamb) else self.lamb[i] for i in range(self.args.total_tasks)]
        # print('lambda tasks', self.lamb)
        self.soft = torch.nn.Softmax(dim=1)
        # self.device = 'cpu'

    def forward(self, inputs, t=None, mode='ets_kbts_cal_ba'):
        cal = False
        if 'cal' in mode:
            cal = True
        if t is not None:
            x = self.dataset.test_transforms[t](inputs)
            outputs = []
            if 'ets' in mode:
                out = self.net.ets_forward(x, t)
                outputs.append(out)
            if 'kbts' in mode:
                out = self.net.kbts_forward(x, t)
                outputs.append(out)

            outputs = ensemble_outputs(outputs)
            _, predicts = outputs.max(1)
            return predicts + t * self.dataset.N_CLASSES_PER_TASK
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            for i in range(self.task+1):
                if 'ba' in mode:
                    # batch augmentation
                    N = self.args.num_aug
                    aug_inputs = inputs.unsqueeze(0).expand(N, *inputs.shape).reshape(N*inputs.shape[0], *inputs.shape[1:])
                    x = self.dataset.train_transform(aug_inputs)
                    x = self.dataset.test_transforms[i](x)
                    x = torch.cat([self.dataset.test_transforms[i](inputs), x])
                else:
                    x = self.dataset.test_transforms[i](inputs)

                if not cal:
                    outputs = []
                    if 'ets' in mode:
                        out = self.net.ets_forward(x, i, cal=cal)
                        outputs.append(out)
                    if 'kbts' in mode:
                        out = self.net.kbts_forward(x, i, cal=cal)
                        outputs.append(out)

                    outputs = ensemble_outputs(outputs)
                else:
                    outputs = self.net.cal_forward(x, t, cal=True)
                
                joint_entropy = entropy(outputs.exp())

                if 'ba' in mode:
                    outputs_tasks.append(outputs.view(N+1, inputs.shape[0], -1)[0])
                    joint_entropy_tasks.append(joint_entropy.view(N+1, inputs.shape[0]).mean(0))
                else:
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)
            
            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, dim=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            _, predicts = predicted_outputs.max(1)
            # print(outputs_tasks.shape, outputs_tasks.abs().sum((0,2)))
            # print('entropy', joint_entropy_tasks.mean((0)))
            # print('mean', outputs_tasks.mean((0)).mean(-1), 'std', outputs_tasks.std((0)).mean(-1))
            # outputs_tasks = outputs_tasks.permute((1, 0, 2)).reshape((self.task+1, -1))
            # print('min - max', outputs_tasks.min(1)[0], outputs_tasks.max(1)[0])
            return predicts + predicted_task * self.dataset.N_CLASSES_PER_TASK
        
    def evaluate(self, task=None, mode='ets_kbts_cal'):
        with torch.no_grad():
            self.net.eval()
            accs = []
            for k, test_loader in enumerate(self.dataset.test_loaders):
                if task is not None:
                    if k not in task:
                        continue
                correct, total = 0.0, 0.0
                for data in test_loader:
                    with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        if task is None:
                            pred = self.forward(inputs, None, mode)
                        else:
                            pred = self.forward(inputs, k, mode)
                        correct += torch.sum(pred == labels).item()
                        total += labels.shape[0]

                acc = correct / total * 100 if 'class-il' in self.COMPATIBILITY else 0
                accs.append(round(acc, 2))

            # model.net.train(status)
            return accs
    
    def train(self, train_loader, progress_bar, mode, squeeze, augment, epoch, verbose=False):
        total = 0
        correct = 0
        total_loss = 0
        if verbose:
            test_acc = self.evaluate([self.task], mode=mode)[0]
            num_params, num_neurons = self.net.count_params()
            num_params = sum(num_params)
        else:
            test_acc = 0
            num_params = 0
        self.net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if augment:
                inputs = self.dataset.train_transform(inputs)
            inputs = self.dataset.test_transforms[self.task](inputs)
            self.opt.zero_grad()
            if mode == 'ets':
                outputs = self.net.ets_forward(inputs, self.task)
            elif mode == 'kbts':
                outputs = self.net.kbts_forward(inputs, self.task)

            loss = self.loss(outputs, labels - self.task * self.dataset.N_CLASSES_PER_TASK)
            loss.backward()
            self.opt.step()
            _, predicts = outputs.max(1)
            correct += torch.sum(predicts == (labels - self.task * self.dataset.N_CLASSES_PER_TASK)).item()
            total += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]
            if squeeze:
                self.net.proximal_gradient_descent(self.scheduler.get_last_lr()[0], self.lamb * self.factor)
                num_neurons = [m.mask_out.sum().item() for m in self.net.DB]
                if verbose:
                    progress_bar.prog(i, len(train_loader), epoch, self.task, total_loss/total, correct/total*100, test_acc, num_params, num_neurons)
            else:
                if verbose:
                    progress_bar.prog(i, len(train_loader), epoch, self.task, total_loss/total, correct/total*100, test_acc, num_params)
        if squeeze:
            self.net.squeeze(self.opt.state)
        self.scheduler.step()

    def train_contrast(self, progress_bar, epoch, mode, verbose=False):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0

        for i, data in enumerate(self.buffer):
            self.opt.zero_grad()
            data = [tmp.to(self.device) for tmp in data]

            # labels = torch.cat([data[2] + t * (self.task+1) for t in range(self.task+1)])
            # labels = torch.cat([(data[2] == t) for t in range(self.task+1)])

            tasks = torch.cat([data[2], data[2]])
            labels = torch.cat([(tasks == t) * (tasks + 1) for t in range(self.task+1)])
            inputs = torch.cat([self.dataset.train_transform(data[0]), self.dataset.train_transform(data[0])])
            features = torch.cat([self.net.cal_forward(self.dataset.test_transforms[t](inputs), t, cal=False) 
                                      for t in range(self.task+1)])
            
            # if 'ets' in mode:
            #     # features = torch.cat([self.net.ets_cal_forward(data[3*t+3], t, cal=False) for t in range(self.task+1)])
            #     features = torch.cat([self.net.cal_forward(self.dataset.test_transforms[t](inputs), t, feat=True) 
            #                           for t in range(self.task+1)])
            # elif 'kbts' in mode:
            #     # features = torch.cat([self.net.kbts_cal_forward(data[3*t+1+3], t, cal=False) for t in range(self.task+1)])
            #     features = torch.cat([self.net.cal_forward(self.dataset.test_transforms[t](inputs), t, feat=True)
            #                           for t in range(self.task+1)])


            # inputs = torch.cat([self.dataset.train_transform(data[0]), self.dataset.train_transform(data[0])])
            # features = self.net.task_feature_layers(inputs)
            # labels = torch.cat([data[2], data[2]])

            loss = sup_con_loss(features, labels, self.args.temperature)
                
            loss.backward()
            self.opt.step()
            total += data[1].shape[0]
            total_loss += loss.item()
            if verbose:
                progress_bar.prog(i, len(self.buffer), epoch, self.task, total_loss/total)

        self.scheduler.step()

    def train_calibration(self, progress_bar, epoch, mode, verbose=False):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0

        for i, data in enumerate(self.buffer):
            self.opt.zero_grad()
            data = [tmp.to(self.device) for tmp in data]
            inputs = self.dataset.train_transform(data[0])

            # outputs = []
            # if 'ets' in mode:
            #     outputs += [torch.cat([self.net.ets_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) 
            #                           for t in range(self.task+1)])]

            # if 'kbts' in mode:
            #     outputs += [torch.cat([self.net.kbts_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) 
            #                           for t in range(self.task+1)])]

            outputs = torch.cat([self.net.cal_forward(self.dataset.test_transforms[t](inputs), t, cal=True) 
                                      for t in range(self.task+1)])
            
            # outputs = ensemble_outputs(outputs)
            join_entropy = entropy(outputs.exp())
            join_entropy = join_entropy.view(self.task+1, data[0].shape[0]).permute(1, 0) # shape [batch size, num tasks]
            labels = torch.stack([(data[2] == t).float() for t in range(self.task+1)], dim=1)
            loss = torch.sum(join_entropy * labels, dim=1) / torch.sum(join_entropy, dim=1)
            loss = torch.mean(loss)

            loss.backward()
            self.opt.step()
            total += data[1].shape[0]
            total_loss += loss.item()
            if verbose:
                progress_bar.prog(i, len(self.buffer), epoch, self.task, total_loss/total)

        self.scheduler.step()


    def begin_task(self, dataset):
        self.task += 1
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for m in self.net.DM:
            # m.kbts_sparsities = torch.cat([m.kbts_sparsities, torch.IntTensor([m.sparsity]).to(device)])
            m.kbts_sparsities += [m.sparsity]
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=0, momentum=self.args.optim_mom)

    def end_task(self, dataset) -> None:
        self.net.freeze()

    def get_rehearsal_logits(self, train_loader):
        if self.task == 0:
            samples_per_task = self.args.buffer_size // (self.task+1)
        else:
            samples_per_task = self.args.buffer_size // (self.task)

        if self.task == 0:
            samples_per_class = self.args.buffer_size // ((self.task+1) * self.dataset.N_CLASSES_PER_TASK)
        else:
            samples_per_class = self.args.buffer_size // (self.task * self.dataset.N_CLASSES_PER_TASK)

        self.net.eval()
        data = [[] for _ in range(3 + (self.task+1) * 3)]

        for inputs, labels in train_loader:
            data[0].append(inputs)
            data[1].append(labels)
            data[2].append(torch.ones_like(labels) * self.task)
            inputs = inputs.to(self.device)
            for i in range(self.task+1):
                x = self.dataset.test_transforms[i](inputs)
                outputs = []
                feat, out = self.net.ets_forward(x, i, feat=True)
                data[3*i+3].append(feat.detach().clone().cpu())
                outputs.append(out)
                feat, out = self.net.kbts_forward(x, i, feat=True)
                data[3*i+1+3].append(feat.detach().clone().cpu())
                outputs.append(out)
                outputs = ensemble_outputs(outputs)
                data[3*i+2+3].append(entropy(outputs.exp()).detach().clone().cpu())

        
        data = [torch.cat(temp) for temp in data]

        if 'be' not in self.args.ablation:
            # buffer entropy
            indices = []
            for c in data[1].unique():
                idx = (data[1] == c)
                if self.task == 0:
                    loss = data[3*self.task+2+3][idx]
                else:
                    join_entropy = torch.stack([data[3*t+2+3][idx] for t in range(self.task+1)], dim=1)
                    labels = torch.stack([(data[2][idx] == t).float() for t in range(self.task+1)], dim=1)
                    loss = torch.sum(join_entropy * labels, dim=1) / torch.sum(join_entropy, dim=1)

                values, stt = loss.sort(dim=0, descending=False)
                indices.append(torch.arange(data[1].shape[0])[idx][stt[:samples_per_class]])
            indices = torch.cat(indices)
            data = [temp[indices] for temp in data]
        else:
            # random class balanced selection
            indices = []
            for c in data[1].unique():
                idx = torch.arange(data[1].shape[0])[data[1] == c][:samples_per_class]
                indices.append(idx)
            indices = torch.cat(indices)
            data = [temp[indices] for temp in data]

        if self.task > 0:
            buf_ent = []
            buf_ets_feat = []
            buf_kbts_feat = []
            for temp in self.buffer:
                inputs = temp[0].to(self.device)
                x = self.dataset.test_transforms[self.task](inputs)
                outputs = []
                feat, out = self.net.ets_forward(x, self.task, feat=True)
                buf_ets_feat.append(feat.detach().clone().cpu())
                outputs.append(out)
                feat, out = self.net.kbts_forward(x, self.task, feat=True)
                buf_kbts_feat.append(feat.detach().clone().cpu())
                outputs.append(out)
                outputs = ensemble_outputs(outputs)
                buf_ent.append(entropy(outputs.exp()).detach().clone().cpu())

            buf_data = list(self.buffer.dataset.tensors) + [torch.cat(buf_ets_feat), torch.cat(buf_kbts_feat), torch.cat(buf_ent)] 
            data = [torch.cat([buf_temp, temp]) for buf_temp, temp in zip(buf_data, data)]
            
        self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        print(data[2].unique())
        print(data[0].shape)
        print(data[1].unique())
        for c in data[1].unique():
            idx = (data[1] == c)
            print(f'{c}: {idx.sum()}', end=', ')
        print()
        


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
        samples_per_task = self.args.buffer_size // (self.task+1)
        samples_per_class = self.args.buffer_size // ((self.task+1) * self.dataset.N_CLASSES_PER_TASK)

        data = list(self.buffer.dataset.tensors)

        if 'be' not in self.args.ablation:
            # buffer entropy
            indices = []
            for c in data[1].unique():
                idx = (data[1] == c)
                if self.task == 0:
                    loss = data[3*self.task+2+3][idx]
                else:
                    join_entropy = torch.stack([data[3*t+2+3][idx] for t in range(self.task+1)], dim=1)
                    labels = torch.stack([(data[2][idx] == t).float() for t in range(self.task+1)], dim=1)
                    loss = torch.sum(join_entropy * labels, dim=1) / torch.sum(join_entropy, dim=1)

                values, stt = loss.sort(dim=0, descending=False)
                indices.append(torch.arange(data[1].shape[0])[idx][stt[:samples_per_class]])
            indices = torch.cat(indices)
            data = [temp[indices] for temp in data]
        else:
            # random class balanced selection
            indices = []
            for c in data[1].unique():
                idx = torch.arange(data[1].shape[0])[data[1] == c][:samples_per_class]
                indices.append(idx)
            indices = torch.cat(indices)
            data = [temp[indices] for temp in data]

        # data = [torch.cat(temp) for temp in data]
        self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        print(data[2].unique())
        print(data[1].unique())
        print(data[0].shape)
        for c in data[1].unique():
            idx = (data[1] == c)
            print(f'{c}: {idx.sum()}', end=', ')
        print()

