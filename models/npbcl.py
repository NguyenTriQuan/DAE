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
from backbone.ResNet18_NPBCL import resnet18, resnet10
from torch.utils.data import DataLoader, Dataset, TensorDataset
from itertools import cycle
import numpy as np
import random
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Dynamic Architecture and Ensemble of Knowledge Base.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--lamb', type=float, required=False,
                        help='regularization factor', default=0)
    parser.add_argument('--alpha', type=float, required=False,
                        help='node-path balance', default=0)
    parser.add_argument('--beta', type=float, required=False,
                        help='kernel induce', default=0)
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
    # outputs shape [num_member, bs, num_cls]
    outputs = F.log_softmax(outputs, dim=-1)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=0)
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

def logsumexp(x, dim=None, keepdim=False):
    """Stable computation of log(sum(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.sum(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def NPB_layer_count(m, mode, t):
    heuristic = 1
    if mode == 'stable':
        mask = m.stable_masks[t]
        if t > 0:
            heuristic = 1 - m.unused_weight + 1
            heuristic = heuristic * heuristic.numel() / heuristic.sum()
    elif mode == 'plastic':
        mask = m.plastic_masks[t]
        if t > 0:
            heuristic = m.unused_weight + 1
            heuristic = heuristic * heuristic.numel() / heuristic.sum()

    if len(m.prev_layers) > 0:
        P_in = torch.logsumexp(torch.stack([n.P_out for n in m.prev_layers], dim=0), dim=0)
        eff_nodes_in, _ = torch.max(torch.stack([n.eff_nodes_out for n in m.prev_layers], dim=0), dim=0)
    else:
        P_in = torch.tensor(0).float().cuda()
        eff_nodes_in = torch.tensor(1).float().cuda()

    m.P_out = torch.logsumexp(mask * heuristic * P_in.view(m.view_in), dim=m.dim_out)
    m.eff_nodes_in = torch.clamp(torch.sum(mask, dim=m.dim_in) * eff_nodes_in, max=1)
    m.eff_nodes_out = torch.clamp(torch.sum(mask * eff_nodes_in.view(m.view_in), dim=m.dim_out), max=1)
    if len(m.weight.shape) == 4:
        m.eff_kernels = torch.clamp(mask.sum(dim=(2,3)), max=1).sum()

def NPB_model_count(net, mode, t, alpha, beta):
    eff_nodes = 0
    eff_kernels = 0
    eff_paths = 0    
    for m in net.DM:
        NPB_layer_count(m, mode, t)
    for m in net.DM:
        if len(m.next_layers) > 0:
            eff_nodes_out, _ = torch.max(torch.stack([m.eff_nodes_out * n.eff_nodes_in for n in m.next_layers], dim=0), dim=0)
        else:
            eff_nodes_out = m.eff_nodes_out
        
        if len(m.weight.shape) == 4:
            eff_kernels += m.eff_kernels

        eff_nodes += eff_nodes_out.sum()

    eff_paths = torch.logsumexp(net.DM[-1].P_out, dim=0)
    eff_nodes = eff_nodes.log()
    eff_kernels = eff_kernels.log()
    for m in net.DM:
        m.P_out = 0
        m.eff_nodes_in = 0
        m.eff_nodes_out = 0
        m.eff_kernels = 0
    return eff_nodes * alpha + eff_paths * (1-alpha) + eff_kernels * beta

class NPBCL(ContinualModel):
    NAME = 'NPBCL'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, dataset):
        super(NPBCL, self).__init__(backbone, loss, args, dataset)

        if args.debug:
            self.net = resnet10(self.dataset.N_CLASSES_PER_TASK, nf=32, args=args).to(device)
        else:
            self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, nf=64, args=args).to(device)
        self.task = -1
        self.buffer = None
        get_related_layers(self.net, self.dataset.INPUT_SHAPE)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.lamb = self.args.lamb

    def forward(self, inputs, t=None, mode='ensemble'):
        if 'ensemble' in mode:
            self.net.set_mode('ensemble')
        elif 'stable' in mode:
            self.net.set_mode('stable')
        elif 'plastic' in mode:
            self.net.set_mode('plastic')
        cal = False
        if 'cal' in mode:
            cal = True
        if 'ba' in mode:
            # batch augmentation
            N = self.args.num_aug 
            x = inputs.unsqueeze(0).expand(N, *inputs.shape).reshape(N*inputs.shape[0], *inputs.shape[1:])
            if 'ensemble' in mode:
                bs = x.shape[0]
                x = torch.cat([x, x], dim=0)
            x = self.dataset.train_transform(x)
        else:
            if 'ensemble' in mode:
                bs = inputs.shape[0]
                x = torch.cat([inputs, inputs], dim=0)
            else:
                x = inputs

        if t is not None:
            outputs = self.net(x, t)
            if 'ensemble' in mode:
                outputs = outputs.split(bs)
            else:
                outputs = [outputs]
            if 'ba' in mode:
                outputs = [out.view(N, inputs.shape[0], -1) for out in outputs]
                outputs = torch.cat(outputs, dim=0)
                # outputs = outputs[:, :, 1:] # ignore ood class
                outputs = ensemble_outputs(outputs)
            else:
                outputs = torch.stack(outputs, dim=0)
                # outputs = outputs[:, :, 1:] # ignore ood class
                outputs = ensemble_outputs(outputs)

            _, predicts = outputs.max(1)
            return predicts + t * (self.dataset.N_CLASSES_PER_TASK)
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            for i in range(self.task+1):
                outputs = self.net(x, i)
                if 'ensemble' in mode:
                    outputs = outputs.split(bs)
                else:
                    outputs = [outputs]

                if 'ba' in mode:
                    outputs = [out.view(N, inputs.shape[0], -1) for out in outputs]
                    outputs = torch.cat(outputs, dim=0)
                    # outputs = outputs[:, :, 1:] # ignore ood class
                    outputs = ensemble_outputs(outputs)
                    joint_entropy = entropy(outputs.exp())
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)
                    # outputs_tasks.append(outputs.view(N+1, inputs.shape[0], -1)[0])
                    # joint_entropy_tasks.append(joint_entropy.view(N+1, inputs.shape[0]).mean(0))
                else:
                    outputs = torch.stack(outputs, dim=0)
                    # outputs = outputs[:, :, 1:] # ignore ood class
                    outputs = ensemble_outputs(outputs)
                    joint_entropy = entropy(outputs.exp())
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
            return predicts + predicted_task * (self.dataset.N_CLASSES_PER_TASK)
        
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
        self.net.train()    
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels - self.task * self.dataset.N_CLASSES_PER_TASK

            # ood_inputs = torch.rot90(inputs, 2, dims=(2,3))
            # ood_labels = torch.zeros_like(labels)
            # if self.buffer is not None:
            #     try:
            #         buffer_data = next(buffer)
            #     except StopIteration:
            #         # restart the generator if the previous generator is exhausted.
            #         buffer = iter(self.buffer)
            #         buffer_data = next(buffer)
            #     buffer_data = [tmp.to(self.device) for tmp in buffer_data]
            #     ood_inputs = torch.cat([ood_inputs, buffer_data[0]], dim=0)
            #     ood_labels = torch.cat([ood_labels, torch.zeros_like(buffer_data[1])], dim=0)

            # inputs = torch.cat([inputs, ood_inputs], dim=0)
            # labels = torch.cat([labels, ood_labels], dim=0)
            inputs = torch.cat([inputs, inputs], dim=0)
            if augment:
                inputs = self.dataset.train_transform(inputs)
            inputs = self.dataset.test_transforms[self.task](inputs)
            self.opt.zero_grad()
            outputs = self.net(inputs, self.task)
            outputs = outputs.split(bs)
            loss = self.loss(outputs[0], labels) + self.loss(outputs[1], labels)
            if 'npb' not in self.args.ablation:
                npb_reg = NPB_model_count(self.net, 'stable', self.task, self.alpha, self.beta) 
                npb_reg += NPB_model_count(self.net, 'plastic', self.task, self.alpha, self.beta)
                loss = loss - self.lamb * npb_reg

            assert not math.isnan(loss)
            loss.backward()
            if self.task > 0:
                self.net.freeze_used_weights()
            self.opt.step()
            outputs = ensemble_outputs(torch.stack(outputs, dim=0))
            _, predicts = outputs.max(1)
            correct += torch.sum(predicts == labels).item()
            total += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]

        self.scheduler.step()
        return total_loss/total, correct/total*100

    
    def train_calibration(self, progress_bar, epoch, mode, verbose=False):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0

        for i, data in enumerate(self.buffer):
            self.opt.zero_grad()
            data = [tmp.to(self.device) for tmp in data]
            inputs = self.dataset.train_transform(data[0])

            outputs = []
            if 'ets' in mode:
                outputs += [torch.cat([self.net.ets_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True)
                                        for t in range(self.task+1)])]

            if 'kbts' in mode:
                outputs += [torch.cat([self.net.kbts_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True)
                                        for t in range(self.task+1)])]

            # outputs = torch.cat([self.net.cal_forward(self.dataset.test_transforms[t](inputs), t, cal=True) 
            #                           for t in range(self.task+1)])
            
            outputs = ensemble_outputs(torch.stack(outputs, dim=0))
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

    def end_task(self, dataset) -> None:
        self.net.update_unused_weights(self.task)

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
                outputs = ensemble_outputs(torch.stack(outputs, dim=0))
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
                # loss = data[3*self.task+2+3][idx]
                values, stt = loss.sort(dim=0, descending=True)
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
                outputs = ensemble_outputs(torch.stack(outputs, dim=0))
                buf_ent.append(entropy(outputs.exp()).detach().clone().cpu())

            buf_data = list(self.buffer.dataset.tensors) + [torch.cat(buf_ets_feat), torch.cat(buf_kbts_feat), torch.cat(buf_ent)] 
            data = [torch.cat([buf_temp, temp]) for buf_temp, temp in zip(buf_data, data)]
            
        self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        # print(data[2].unique())
        # print(data[0].shape)
        # print(data[1].unique())
        # for c in data[1].unique():
        #     idx = (data[1] == c)
        #     print(f'{c}: {idx.sum()}', end=', ')
        # print()
        


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
                # loss = data[3*self.task+2+3][idx]
                values, stt = loss.sort(dim=0, descending=True)
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
        # print(data[2].unique())
        # print(data[1].unique())
        # print(data[0].shape)
        # for c in data[1].unique():
        #     idx = (data[1] == c)
        #     print(f'{c}: {idx.sum()}', end=', ')
        # print()

def get_related_layers(net, input_shape):
    def stable_check(x1, x2):
        eps = x1 * 1e-4
        return abs(x1 - x2) < eps

    def forward_hook(m, i, o):
        m.output_idx = random.random() * m.input_idx
        o[0] += m.output_idx
        return
    
    def forward_pre_hook(m, i):
        if len(i[0].shape) == 4:
            m.input_idx = i[0][0,0,0,0].item()
        else:
            m.input_idx = i[0][0,0].item()
        return (torch.zeros_like(i[0]), i[1])

    handes = []
    for m in net.DM:
        h1 = m.register_forward_pre_hook(forward_pre_hook)
        h2 = m.register_forward_hook(forward_hook)
        handes += [h1, h2]
    c, h, w = input_shape
    data = torch.ones((1, c, h, w), dtype=torch.float, device='cuda')
    net.eval()
    net.set_mode('stable')
    out = net(data, 0)  
    for h in handes:
        h.remove()
    
    for n, m in net.named_modules():
        if m in net.DM:
            # print(n, m.weight.shape, m.input_idx, m.output_idx)
            m.name = n
            m.prev_layers = []
            m.next_layers = []
    eps = 1e-3
    for i in range(len(net.DM)):
        for j in range(i):
            # if abs(scores[i].input_idx - scores[j].output_idx) < eps:
            if stable_check(net.DM[i].input_idx, net.DM[j].output_idx):
                net.DM[i].prev_layers.append(net.DM[j])
                net.DM[j].next_layers.append(net.DM[i])
            else:
                for k in range(j):
                    # if abs(scores[i].input_idx - scores[j].output_idx - scores[k].output_idx) < eps:
                    if stable_check(net.DM[i].input_idx, net.DM[j].output_idx + net.DM[k].output_idx):
                        net.DM[i].prev_layers += [net.DM[j], net.DM[k]]
                        net.DM[j].next_layers += [net.DM[i]]
                        net.DM[k].next_layers += [net.DM[i]]

                    # if abs(scores[i].input_idx - scores[j].output_idx - scores[k].input_idx) < eps:
                    if stable_check(net.DM[i].input_idx, net.DM[j].output_idx + net.DM[k].input_idx):
                        net.DM[i].prev_layers += [net.DM[j]] + net.DM[k].prev_layers
                        net.DM[j].next_layers += [net.DM[i]]
                        for g in net.DM[k].prev_layers:
                            g.next_layers += [net.DM[i]]

                    # if abs(scores[i].input_idx - scores[k].output_idx - scores[j].input_idx) < eps:
                    if stable_check(net.DM[i].input_idx, net.DM[k].output_idx + net.DM[j].input_idx):
                        net.DM[i].prev_layers += [net.DM[k]] + net.DM[j].prev_layers
                        net.DM[k].next_layers += [net.DM[i]]
                        for g in net.DM[j].prev_layers:
                            g.next_layers += [net.DM[i]]

                    # if abs(scores[i].input_idx - scores[j].input_idx - scores[k].input_idx) < eps:
                    if stable_check(net.DM[i].input_idx, net.DM[j].input_idx + net.DM[k].input_idx):
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
        print(m.name, m.weight.shape, m.input_idx, m.output_idx)
        for j, n in enumerate(m.prev_layers):
            print('prev', j, n.name, n.weight.shape, n.input_idx, n.output_idx)
        for j, n in enumerate(m.next_layers):
            print('next', j, n.name, n.weight.shape, n.input_idx, n.output_idx)
        print()
    
    net.prev_layers = list(set(net.prev_layers))
    for i, layers in enumerate(net.prev_layers):
        if len(layers) == 0:
            print(i, layers)
        for m in layers:
            print(i, m.name)

    all_layers = []
    for layers in net.prev_layers:
        all_layers += list(layers)
    # for m in net.DM[:-1]:
    #     assert m in all_layers