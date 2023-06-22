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
import wandb
from utils.status import ProgressBar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Continual Learning with Dynamic Architecture and Ensemble of Knowledge Base.")

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--lamb", type=str, required=True, help="capacity control.")
    parser.add_argument("--alpha", type=float, required=False, help="maximize entropy of ood samples loss factor.", default=1)
    parser.add_argument("--dropout", type=float, required=False, help="Dropout probability.", default=0.0)
    parser.add_argument("--sparsity", type=float, required=True, help="Super mask sparsity.")
    parser.add_argument("--temperature", default=0.1, type=float, required=False, help="Supervised Contrastive loss temperature.")
    parser.add_argument("--negative_slope", default=0, type=float, required=False, help="leaky relu activation negative slope.")
    parser.add_argument("--ablation", type=str, required=False, help="Ablation study.", default="")
    parser.add_argument("--mode", type=str, required=False, help="Ablation study.", default="")
    parser.add_argument("--norm_type", type=str, required=False, help="batch normalization layer", default="none")
    parser.add_argument("--debug", action="store_true", help="Quick test.")
    parser.add_argument("--verbose", action="store_true", help="compute test accuracy and number of params.")
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--cal", action="store_true", help="calibration training")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--lr_score", type=float, required=False, help="score learning rate.", default=0.1)
    parser.add_argument("--num_tasks", type=int, required=False, help="number of tasks to run.", default=100)
    parser.add_argument("--total_tasks", type=int, required=True, help="total number of tasks.", default=10)
    parser.add_argument("--eps", type=float, required=False, help="FGSM epsilon.", default=0.1)
    parser.add_argument("--num_aug", type=int, required=False, help="number of augument samples used when evaluation.", default=16)
    parser.add_argument("--task", type=int, required=False, help="Specify task for eval or cal.", default=-1)
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


def entropy(x):
    return -torch.sum(x * torch.log(x+1e-9), dim=1)


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
    outputs = torch.stack(outputs, dim=-1)  # [bs, num_cls, num_member]
    weights = torch.stack(weights, dim=-1)  # [bs, num_member]

    weights = F.softmax(weights / temperature, dim=-1).unsqueeze(1)  # [bs, 1, num_member]
    outputs = F.log_softmax(outputs, dim=-2)
    output_max, _ = torch.max(outputs, dim=-1, keepdim=True)
    log_outputs = output_max + torch.log(torch.sum((outputs - output_max).exp() * weights, dim=-1, keepdim=True))
    return log_outputs.squeeze(-1)


def sup_clr_ood_loss(ind_features, features, labels, temperature):
    labels = labels.repeat(2)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(ind_features, features.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = (1 - torch.eye(ind_features.shape[0]).to(device))  # remove diagonal shape: num ind, num ind
    mask = mask * logits_mask
    extend_mask = torch.ones(ind_features.shape[0], features.shape[0] - ind_features.shape[0]).to(device)
    logits_mask = torch.cat([logits_mask, extend_mask], dim=1) # shape num ind, num ind + ood
    mask = torch.cat([mask, 1-extend_mask], dim=1)

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss

def sup_clr_loss(features, labels, temperature):
    labels = labels.repeat(2)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = (1 - torch.eye(features.shape[0]).to(device))  # remove diagonal shape: num ind, num ind
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -mean_log_prob_pos
    loss = loss.mean()

    return loss


class DAE(ContinualModel):
    NAME = "DAE"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, dataset):
        super(DAE, self).__init__(backbone, loss, args, dataset)
        if args.norm_type == "none":
            norm_type = None
        else:
            norm_type = args.norm_type

        if args.debug:
            self.net = resnet10(0, norm_type=norm_type, args=args)
            # self.net = resnet18(self.dataset.N_CLASSES_PER_TASK, norm_type=norm_type, args=args)
        else:
            self.net = resnet18(0, norm_type=norm_type, args=args)
        self.task = -1
        # try:
        #     self.lamb = float(args.lamb)
        # except:
        #     self.lamb = [float(i) for i in args.lamb.split('_')][0]
        self.lamb = [float(i) for i in args.lamb.split("_")]
        if len(self.lamb) < self.args.total_tasks:
            self.lamb = [self.lamb[-1] if i >= len(self.lamb) else self.lamb[i] for i in range(self.args.total_tasks)]
        print("lambda tasks", self.lamb)
        self.soft = torch.nn.Softmax(dim=1)
        self.alpha = args.alpha
        self.eps = args.eps
        self.buffer = None

    def forward(self, inputs, t=None, ets=True, kbts=False, cal=True, ba=True):
        bs = inputs.shape[0]
        if ba:
            # batch augmentation
            N = self.args.num_aug
            # aug_inputs = inputs.unsqueeze(0).expand(N, *inputs.shape).reshape(N * inputs.shape[0], *inputs.shape[1:])
            inputs = inputs.repeat(N, 1, 1, 1)
            x = self.dataset.train_transform(inputs)
        else:
            x = inputs

        if t is not None:
            outputs = []
            if ets:
                out = self.net.ets_forward(x, t)
                outputs.append(out)
            if kbts:
                out = self.net.kbts_forward(x, t)
                outputs.append(out)

            if ba:
                outputs = [out.view(N, bs, -1) for out in outputs]
                outputs = torch.cat(outputs, dim=0)
                # outputs = outputs[:, :, 1:]  # ignore ood class
                outputs = ensemble_outputs(outputs)
            else:
                outputs = torch.stack(outputs, dim=0)
                # outputs = outputs[:, :, 1:]  # ignore ood class
                outputs = ensemble_outputs(outputs)

            predicts = outputs.argmax(1)
            del x, outputs
            return predicts + t * (self.dataset.N_CLASSES_PER_TASK)
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            for i in range(self.task + 1):
                outputs = []
                if ets:
                    out = self.net.ets_forward(x, i, cal=cal)
                    outputs.append(out)
                if kbts:
                    out = self.net.kbts_forward(x, i, cal=cal)
                    outputs.append(out)

                if ba:
                    outputs = [out.view(N, bs, -1) for out in outputs]
                    outputs = torch.cat(outputs, dim=0)
                    # outputs = outputs[:, :, 1:]  # ignore ood class
                    outputs = ensemble_outputs(outputs)
                    joint_entropy = entropy(outputs.exp())
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)
                else:
                    outputs = torch.stack(outputs, dim=0)
                    # outputs = outputs[:, :, 1:]  # ignore ood class
                    outputs = ensemble_outputs(outputs)
                    joint_entropy = entropy(outputs.exp())
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)

            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, dim=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            cil_predicts = predicted_outputs.argmax(1)
            cil_predicts = cil_predicts + predicted_task * (self.dataset.N_CLASSES_PER_TASK)
            del x, joint_entropy_tasks, predicted_outputs
            return cil_predicts, outputs_tasks, predicted_task

    def evaluate(self, task=None, mode="ets_kbts_cal_ba"):
        kbts = "kbts" in mode
        ets = "ets" in mode
        cal = "cal" in mode
        ba = "ba" in mode

        with torch.no_grad():
            self.net.eval()
            til_accs = []
            cil_accs = []
            task_correct = 0
            task_total = 0
            for k, test_loader in enumerate(self.dataset.test_loaders):
                if task is not None:
                    if k != task:
                        continue
                cil_correct, til_correct, total = 0.0, 0.0, 0.0
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if task is None:
                        cil_predicts, outputs, predicted_task = self.forward(inputs, None, ets, kbts, cal, ba)
                        cil_correct += torch.sum(cil_predicts == labels).item()
                        til_predicts = outputs[:, k].argmax(1) + k * (self.dataset.N_CLASSES_PER_TASK)
                        til_correct += torch.sum(til_predicts == labels).item()
                        task_correct += torch.sum(predicted_task == k).item()
                        total += labels.shape[0]
                        del cil_predicts, outputs, predicted_task
                    else:
                        til_predicts = self.forward(inputs, task, ets, kbts, cal, ba)
                        til_correct += torch.sum(til_predicts == labels).item()
                        total += labels.shape[0]
                        del til_predicts

                til_accs.append(round(til_correct / total * 100, 2))
                cil_accs.append(round(cil_correct / total * 100, 2))
                task_total += total
            if task is None:
                task_acc = round(task_correct / task_total * 100, 2)
                cil_avg = round(np.mean(cil_accs), 2)
                til_avg = round(np.mean(til_accs), 2)
                print(f"Task {len(til_accs)-1}: {mode}: cil {cil_avg} {cil_accs}, til {til_avg} {til_accs}, tp {task_acc}")
                if self.args.verbose:
                    wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, f"{mode}_tp": task_acc, 'task':len(til_accs) - 1})
                return til_accs, cil_accs, task_acc
            else:
                return til_accs[0]

    def train_contrast(self, train_loader, mode, ets, kbts, rot, buf, adv, squeeze, augment):
        total = 0
        correct = 0
        total_loss = 0

        self.net.train()
        ood = buf or rot

        if self.buffer is not None:
            buffer = iter(self.buffer)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels - self.task * self.dataset.N_CLASSES_PER_TASK
            ood_inputs = torch.empty(0).to(self.device)
            if adv:
                inputs.requires_grad = True
                adv_inputs = fgsm_attack(inputs, self.eps, inputs_grad)
                ood_inputs = torch.cat([ood_inputs, buffer_data[0]], dim=0)
            if rot:
                rot = random.randint(1, 3)
                ood_inputs = torch.cat([ood_inputs, torch.rot90(inputs, rot, dims=(2, 3))], dim=0)
            if buf:
                if self.buffer is not None:
                    try:
                        buffer_data = next(buffer)
                    except StopIteration:
                        # restart the generator if the previous generator is exhausted.
                        buffer = iter(self.buffer)
                        buffer_data = next(buffer)
                    buffer_data = [tmp.to(self.device) for tmp in buffer_data]
                    ood_inputs = torch.cat([ood_inputs, buffer_data[0]], dim=0)

            # if feat:
            #     inputs = torch.cat([inputs, inputs], dim=0)
            if ood:
                inputs = torch.cat([inputs, ood_inputs], dim=0)
            
            if augment:
                inputs = self.dataset.train_transform(inputs)
            # inputs = self.dataset.test_transforms[self.task](inputs)
            self.opt.zero_grad()
            if ets:
                outputs = self.net.ets_forward(inputs, self.task, feat=False)
            elif kbts:
                outputs = self.net.kbts_forward(inputs, self.task, feat=False)

            # if feat:
            #     outputs = F.normalize(outputs, p=2, dim=1)
            #     if ood:
            #         ind_outputs = outputs[:bs*2]
            #         loss = sup_clr_ood_loss(ind_outputs, outputs, labels, self.args.temperature)
            #     else:
            #         loss = sup_clr_loss(outputs, labels, self.args.temperature)
            # else:
            if ood and (ood_inputs.numel() > 0):
                ind_outputs = outputs[:bs]
                ood_outputs = outputs[bs:]
                # loss = (self.loss(ind_outputs, labels) + self.loss(ood_outputs, ood_labels)) / 2
                ood_outputs = ensemble_outputs(ood_outputs.unsqueeze(0))
                loss = self.loss(ind_outputs, labels) - self.alpha * entropy(ood_outputs.exp()).mean()
                outputs = ind_outputs
                # loss = self.loss(ind_outputs, labels) / (entropy(ood_outputs.exp()).mean()+1e-9)
            else:
                loss = self.loss(outputs, labels)
            assert not math.isnan(loss)
            loss.backward()
            self.opt.step()
            total += bs
            total_loss += loss.item() * bs
            correct += (outputs.argmax(1) == labels).sum().item()
            if squeeze:
                self.net.proximal_gradient_descent(self.scheduler.get_last_lr()[0], self.lamb[self.task])
                
        if squeeze:
            self.net.squeeze(self.opt.state)
        self.scheduler.step()

        return total_loss / total, round(correct * 100 / total, 2)

    def train_calibration(self, mode, ets, kbts):
        self.net.train()
        total = 0
        correct = 0
        total_loss = 0

        for i, data in enumerate(self.buffer):
            self.opt.zero_grad()
            data = [tmp.to(self.device) for tmp in data]
            inputs = self.dataset.train_transform(data[0])

            outputs = []
            if ets:
                outputs += [torch.cat([self.net.ets_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) for t in range(self.task + 1)])]

            if kbts:
                outputs += [torch.cat([self.net.kbts_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) for t in range(self.task + 1)])]

            outputs = torch.stack(outputs, dim=0)
            # outputs = outputs[:, :, 1:]  # ignore ood class
            outputs = ensemble_outputs(outputs)
            join_entropy = entropy(outputs.exp())
            join_entropy = join_entropy.view(self.task + 1, data[0].shape[0]).permute(1, 0)  # shape [batch size, num tasks]
            labels = torch.stack([(data[2] == t).float() for t in range(self.task + 1)], dim=1)
            loss = torch.sum(join_entropy * labels, dim=1) / (torch.sum(join_entropy, dim=1)+1e-9)
            # loss = - torch.sum(join_entropy * (1-labels), dim=1)
            loss = torch.mean(loss)

            assert not math.isnan(loss)
            loss.backward()
            self.opt.step()
            total += data[1].shape[0]
            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / total

    def begin_task(self, dataset):
        self.task += 1
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for m in self.net.DM:
            m.kbts_sparsities += [m.sparsity]

    def end_task(self, dataset) -> None:
        self.net.freeze_feature()
        self.net.freeze_classifier()
        self.net.clear_memory()

    def get_rehearsal_logits(self, train_loader):
        if self.task == 0:
            samples_per_task = self.args.buffer_size // (self.task + 1)
        else:
            samples_per_task = self.args.buffer_size // (self.task)

        if self.task == 0:
            samples_per_class = self.args.buffer_size // ((self.task + 1) * self.dataset.N_CLASSES_PER_TASK)
        else:
            samples_per_class = self.args.buffer_size // (self.task * self.dataset.N_CLASSES_PER_TASK)

        self.net.eval()
        # data = [[] for _ in range(3 + (self.task+1) * 3)]
        data = [[] for _ in range(3)]

        for inputs, labels in train_loader:
            data[0].append(inputs)
            data[1].append(labels)
            data[2].append(torch.ones_like(labels) * self.task)
            # inputs = inputs.to(self.device)
            # for i in range(self.task+1):
            #     x = self.dataset.test_transforms[i](inputs)
            #     outputs = []
            #     feat, out = self.net.ets_forward(x, i, feat=True)
            #     data[3*i+3].append(feat.detach().clone().cpu())
            #     outputs.append(out)
            #     feat, out = self.net.kbts_forward(x, i, feat=True)
            #     data[3*i+1+3].append(feat.detach().clone().cpu())
            #     outputs.append(out)
            #     outputs = ensemble_outputs(torch.stack(outputs, dim=0))
            #     data[3*i+2+3].append(entropy(outputs.exp()).detach().clone().cpu())

        data = [torch.cat(temp) for temp in data]

        # if 'be' not in self.args.ablation:
        #     # buffer entropy
        #     indices = []
        #     for c in data[1].unique():
        #         idx = (data[1] == c)
        #         if self.task == 0:
        #             loss = data[3*self.task+2+3][idx]
        #         else:
        #             join_entropy = torch.stack([data[3*t+2+3][idx] for t in range(self.task+1)], dim=1)
        #             labels = torch.stack([(data[2][idx] == t).float() for t in range(self.task+1)], dim=1)
        #             loss = torch.sum(join_entropy * labels, dim=1) / torch.sum(join_entropy, dim=1)
        #         # loss = data[3*self.task+2+3][idx]
        #         values, stt = loss.sort(dim=0, descending=True)
        #         indices.append(torch.arange(data[1].shape[0])[idx][stt[:samples_per_class]])
        #     indices = torch.cat(indices)
        #     data = [temp[indices] for temp in data]
        # else:
        # random class balanced selection
        indices = []
        for c in data[1].unique():
            idx = torch.arange(data[1].shape[0])[data[1] == c][:samples_per_class]
            indices.append(idx)
        indices = torch.cat(indices)
        data = [temp[indices] for temp in data]

        if self.task > 0:
            # buf_ent = []
            # buf_ets_feat = []
            # buf_kbts_feat = []
            # for temp in self.buffer:
            #     inputs = temp[0].to(self.device)
            #     x = self.dataset.test_transforms[self.task](inputs)
            #     outputs = []
            #     feat, out = self.net.ets_forward(x, self.task, feat=True)
            #     buf_ets_feat.append(feat.detach().clone().cpu())
            #     outputs.append(out)
            #     feat, out = self.net.kbts_forward(x, self.task, feat=True)
            #     buf_kbts_feat.append(feat.detach().clone().cpu())
            #     outputs.append(out)
            #     outputs = ensemble_outputs(torch.stack(outputs, dim=0))
            #     buf_ent.append(entropy(outputs.exp()).detach().clone().cpu())

            # buf_data = list(self.buffer.dataset.tensors) + [torch.cat(buf_ets_feat), torch.cat(buf_kbts_feat), torch.cat(buf_ent)]
            buf_data = list(self.buffer.dataset.tensors)
            data = [torch.cat([buf_temp, temp]) for buf_temp, temp in zip(buf_data, data)]

        self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        # print(data[2].unique())
        # print(data[0].shape)
        # print(data[1].unique())
        for c in data[1].unique():
            idx = data[1] == c
            print(f"{c}: {idx.sum()}", end=", ")
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
        samples_per_task = self.args.buffer_size // (self.task + 1)
        samples_per_class = self.args.buffer_size // ((self.task + 1) * self.dataset.N_CLASSES_PER_TASK)

        data = list(self.buffer.dataset.tensors)

        # if 'be' not in self.args.ablation:
        #     # buffer entropy
        #     indices = []
        #     for c in data[1].unique():
        #         idx = (data[1] == c)
        #         if self.task == 0:
        #             loss = data[3*self.task+2+3][idx]
        #         else:
        #             join_entropy = torch.stack([data[3*t+2+3][idx] for t in range(self.task+1)], dim=1)
        #             labels = torch.stack([(data[2][idx] == t).float() for t in range(self.task+1)], dim=1)
        #             loss = torch.sum(join_entropy * labels, dim=1) / torch.sum(join_entropy, dim=1)
        #         # loss = data[3*self.task+2+3][idx]
        #         values, stt = loss.sort(dim=0, descending=True)
        #         indices.append(torch.arange(data[1].shape[0])[idx][stt[:samples_per_class]])
        #     indices = torch.cat(indices)
        #     data = [temp[indices] for temp in data]
        # else:
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
        for c in data[1].unique():
            idx = data[1] == c
            print(f"{c}: {idx.sum()}", end=", ")
        print()
