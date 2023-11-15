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
# from utils.buffer import Buffer, icarl_replay
from backbone.ResNet18_MEAE import resnet18, resnet10
from torch.utils.data import DataLoader, Dataset, TensorDataset
from itertools import cycle
from backbone.utils.meae_layers import DynamicLinear, DynamicConv2D, DynamicClassifier, _DynamicLayer, DynamicNorm
import numpy as np
import random
import math
import wandb
from utils.status import ProgressBar
from utils.distributed import make_dp
import copy

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,6,7"


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Continual Learning with Dynamic Architecture and Ensemble of Knowledge Base.")

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--lamb", type=str, required=True, help="capacity control.")
    parser.add_argument("--alpha", type=float, default=1, required=False, help="maximize entropy of ood samples loss factor.")
    parser.add_argument("--beta", type=float, default=1, required=False)
    parser.add_argument("--dropout", type=float, required=False, help="Dropout probability.", default=0.0)
    parser.add_argument("--sparsity", type=float, required=True, help="Super mask sparsity.")
    parser.add_argument("--temperature", default=0.1, type=float, required=False, help="Supervised Contrastive loss temperature.")
    parser.add_argument("--negative_slope", default=0, type=float, required=False, help="leaky relu activation negative slope.")
    parser.add_argument("--ablation", type=str, required=False, help="Ablation study.", default="")
    parser.add_argument("--mode", type=str, required=False, help="Ablation study.", default="")
    parser.add_argument("--norm_type", type=str, required=False, help="batch normalization layer", default="none")
    parser.add_argument("--debug", action="store_true", help="Quick test.")
    parser.add_argument("--verbose", action="store_true", help="compute test accuracy and number of params.")
    parser.add_argument("--wandb", action="store_true", help="wandb")
    parser.add_argument("--amp", action="store_true", help="mix precision")
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--cal", action="store_true", help="calibration training")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--lr_score", type=float, required=False, help="score learning rate.", default=0.1)
    parser.add_argument("--num_tasks", type=int, required=False, help="number of tasks to run.", default=100)
    parser.add_argument("--total_tasks", type=int, required=True, help="total number of tasks.", default=10)
    parser.add_argument("--eps", type=float, required=False, help="FGSM epsilon.", default=5)
    parser.add_argument("--num_aug", type=int, required=False, help="number of augument samples used when evaluation.", default=16)
    parser.add_argument("--task", type=int, required=False, help="Specify task for eval or cal.", default=-1)
    parser.add_argument("--threshold", type=float, required=False, help="GPM threshold.", default=0.97)
    parser.add_argument('--scale', type=float, nargs='*', default=[0.08, 1.0],
                        help='resized crop scale (default: [])', required=False)
    parser.add_argument("--device", type=str, required=False, help="training device: cuda:id or cpu", default="cuda:0")
    parser.add_argument("--gpu_id", type=int, required=False, help="id of the gpu to run on", default=0)
    return parser

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # sign_data_grad = data_grad / data_grad.norm(2)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(targets, outputs):
    return -torch.sum(targets * torch.log(outputs+1e-9), dim=1)


def entropy(x):
    return -torch.sum(x * torch.log(x+1e-9), dim=1)


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def ensemble_outputs(outputs, dim=0):
    # outputs shape [num_member, bs, num_cls]
    outputs = F.log_softmax(outputs, dim=-1)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=dim)
    return log_outputs


def weighted_ensemble(outputs, weights, temperature):
    outputs = torch.stack(outputs, dim=-1)  # [bs, num_cls, num_member]
    weights = torch.stack(weights, dim=-1)  # [bs, num_member]

    weights = F.softmax(weights / temperature, dim=-1).unsqueeze(1)  # [bs, 1, num_member]
    outputs = F.log_softmax(outputs, dim=-2)
    output_max, _ = torch.max(outputs, dim=-1, keepdim=True)
    log_outputs = output_max + torch.log(torch.sum((outputs - output_max).exp() * weights, dim=-1, keepdim=True))
    return log_outputs.squeeze(-1)




class MEAE(ContinualModel):
    NAME = "MEAE"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, dataset):
        super(MEAE, self).__init__(backbone, loss, args, dataset)
        self.device = args.device
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
        self.beta = args.beta
        self.eps = args.eps
        self.buffer = None

    def forward(self, inputs, t=None, ets=True, kbts=False, cal=True, ba=True):
        # torch.cuda.empty_cache()
        B = inputs.shape[0]
        if ba:
            # batch augmentation
            B, N, C, W, H = inputs.shape
            inputs = inputs.view(B*N, C, W, H)

            # N = self.args.num_aug
            # inputs = inputs.repeat(N, 1, 1, 1)
            # inputs = self.dataset.train_transform(inputs)

        inputs = inputs.to(self.device)

        if t is not None:
            outputs = []
            if ets:
                outputs.append(self.net.ets_forward(inputs, t))
            if kbts:
                outputs.append(self.net.kbts_forward(inputs, t))

            if ba:
                outputs = [out.view(B, N, -1) for out in outputs]
                outputs = torch.cat(outputs, dim=1)
                outputs = outputs[:, :, :-1]  # ignore ood class
                outputs = ensemble_outputs(outputs, dim=1)
            else:
                outputs = torch.stack(outputs, dim=1)
                outputs = outputs[:, :, :-1]  # ignore ood class
                outputs = ensemble_outputs(outputs, dim=1)

            out_min = outputs.min(1)[1].view(-1, 1)
            out_max = outputs.max(1)[1].view(-1, 1)
            outputs = (outputs - out_min) / (out_max - out_min)
            outputs = outputs / outputs.sum(1).view(-1,1)

            predicts = outputs.argmax(1)
            del inputs, outputs
            return predicts + t * (self.dataset.N_CLASSES_PER_TASK)
        else:
            joint_entropy_tasks = []
            outputs_tasks = []
            for i in range(self.task + 1):
                if cal:
                    w_ets = self.net.w_ets
                    b_ets = self.net.b_ets
                    w_kbts = self.net.w_kbts
                    b_kbts = self.net.b_kbts
                else: 
                    w_ets = 1
                    b_ets = 0
                    w_kbts = 1
                    b_kbts = 0

                outputs = []
                if ets:
                    outputs.append(self.net.ets_forward(inputs, i, cal=cal) * w_ets + b_ets)
                if kbts:
                    outputs.append(self.net.kbts_forward(inputs, i, cal=cal) * w_kbts + b_kbts)

                if ba:
                    outputs = [out.view(B, N, -1) for out in outputs]
                    outputs = torch.cat(outputs, dim=1)
                    outputs = outputs[:, :, :-1]  # ignore ood class
                    outputs = ensemble_outputs(outputs, dim=1)

                    out_min = outputs.min(1)[1].view(-1, 1)
                    out_max = outputs.max(1)[1].view(-1, 1)
                    outputs = (outputs - out_min) / (out_max - out_min)
                    outputs = outputs / outputs.sum(1).view(-1,1)

                    joint_entropy = entropy(outputs.exp())
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)
                else:
                    outputs = torch.stack(outputs, dim=1)
                    outputs = outputs[:, :, :-1]  # ignore ood class 
                    outputs = ensemble_outputs(outputs, dim=1)

                    out_min = outputs.min(1)[1].view(-1, 1)
                    out_max = outputs.max(1)[1].view(-1, 1)
                    outputs = (outputs - out_min) / (out_max - out_min)
                    outputs = outputs / outputs.sum(1).view(-1,1)

                    joint_entropy = entropy(outputs.exp())
                    outputs_tasks.append(outputs)
                    joint_entropy_tasks.append(joint_entropy)

            outputs_tasks = torch.stack(outputs_tasks, dim=1)
            joint_entropy_tasks = torch.stack(joint_entropy_tasks, dim=1)
            predicted_task = torch.argmin(joint_entropy_tasks, dim=1)
            predicted_outputs = outputs_tasks[range(outputs_tasks.shape[0]), predicted_task]
            cil_predicts = predicted_outputs.argmax(1)
            cil_predicts = cil_predicts + predicted_task * (self.dataset.N_CLASSES_PER_TASK)
            del inputs, joint_entropy_tasks, predicted_outputs
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
                if ba:
                    test_loader.dataset.num_aug = self.args.num_aug
                else: 
                    test_loader.dataset.num_aug = 0

                cil_correct, til_correct, total = 0.0, 0.0, 0.0
                for data in test_loader:
                    inputs, labels = data
                    labels = labels.to(self.device)
                    if task is None:
                        cil_predicts, outputs, predicted_task = self.forward(inputs, None, ets, kbts, cal, ba)
                        cil_correct += torch.sum(cil_predicts == labels).item()
                        til_predicts = outputs[:, k].argmax(1) + k * (self.dataset.N_CLASSES_PER_TASK)
                        til_correct += torch.sum(til_predicts == labels).item()
                        task_correct += torch.sum(predicted_task == k).item()
                        total += labels.shape[0]
                        del cil_predicts, outputs, predicted_task, inputs, labels
                    else:
                        til_predicts = self.forward(inputs, task, ets, kbts, cal, ba)
                        til_correct += torch.sum(til_predicts == labels).item()
                        total += labels.shape[0]
                        del til_predicts, inputs, labels

                til_accs.append(round(til_correct / total * 100, 2))
                cil_accs.append(round(cil_correct / total * 100, 2))
                task_total += total
            if task is None:
                task_acc = round(task_correct / task_total * 100, 2)
                cil_avg = round(np.mean(cil_accs), 2)
                til_avg = round(np.mean(til_accs), 2)
                print(f"Task {len(til_accs)-1}: {mode}: cil {cil_avg} {cil_accs}, til {til_avg} {til_accs}, tp {task_acc}")
                if self.args.verbose:
                    if self.args.wandb:
                        wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, f"{mode}_tp": task_acc, 'task':len(til_accs) - 1})
                return til_accs, cil_accs, task_acc
            else:
                return til_accs[0]
            
    def gen_adv_ood(self, train_loader, ets, kbts):
        self.net.freeze(False)
        self.net.eval()
        all_adv_inputs = []
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels - self.task * self.dataset.N_CLASSES_PER_TASK
            inputs.requires_grad = True
            if ets:
                outputs = self.net.ets_forward(inputs, self.task, feat=False)
            elif kbts:
                outputs = self.net.kbts_forward(inputs, self.task, feat=False)
            outputs = ensemble_outputs(outputs.unsqueeze(0)) 
            self.opt.zero_grad()
            loss = F.nll_loss(outputs, labels, reduce=False) - self.alpha * entropy(outputs.exp())
            grads = []
            for i, l in enumerate(loss):
                grad = torch.autograd.grad(l, inputs, retain_graph=True)[0]
                grads.append(grad[i])
            adv_inputs = fgsm_attack(inputs, self.eps, torch.stack(grads, dim=0))
            if ets:
                outputs = self.net.ets_forward(adv_inputs, self.task, feat=False)
            elif kbts:
                outputs = self.net.kbts_forward(adv_inputs, self.task, feat=False)
            incorrect = outputs.argmax(1) != labels
            all_adv_inputs += [adv_inputs[incorrect].detach().cpu()]
        
        all_adv_inputs = torch.cat(all_adv_inputs, dim=0)
        self.adv_loader = DataLoader(TensorDataset(all_adv_inputs), batch_size=self.args.batch_size, shuffle=True)
        self.net.freeze(True)

    def train_contrast(self, train_loader, mode, ets, kbts, rot, buf, adv, feat, squeeze, augment, kd):
        total = 0
        ets_correct = 0
        ets_total_loss = 0

        kbts_correct = 0
        kbts_total_loss = 0

        enabled = False
        if self.args.amp:
            enabled = True
            torch.backends.cudnn.benchmark = True
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            scaler2 = torch.cuda.amp.GradScaler(enabled=True)

        if self.buffer is not None:
            buffer = iter(self.buffer)
        
        self.net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            # inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels - self.task * self.dataset.N_CLASSES_PER_TASK
            ood_inputs = torch.empty(0)
            num_ood = 0
            if rot:
                rot = random.randint(1, 3)
                ood_inputs = torch.cat([ood_inputs, torch.rot90(inputs, rot, dims=(2, 3))], dim=0)
                num_ood += inputs.shape[0]
            if buf:
                if self.buffer is not None:
                    try:
                        buffer_data, buffer_targets = next(buffer)
                    except StopIteration:
                        # restart the generator if the previous generator is exhausted.
                        buffer = iter(self.buffer)
                        buffer_data, buffer_targets = next(buffer)
                    ood_inputs = torch.cat([ood_inputs, buffer_data], dim=0)
                    num_ood += buffer_data.shape[0]

            inputs = torch.cat([inputs, ood_inputs], dim=0)
            # if augment:
            #     inputs = self.dataset.train_transform(inputs)

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=enabled):
                ets_outputs = self.net.ets_forward(inputs, self.task, feat=False)
                kbts_outputs = self.net.kbts_forward(inputs, self.task, feat=False)
                # print(ets_outputs.mean(0).sum(), ets_outputs.var(0).sum())
                # print(kbts_outputs.mean(0).sum(), kbts_outputs.var(0).sum())
                ets_outputs = F.softmax(ets_outputs, dim=1)
                kbts_outputs = F.softmax(kbts_outputs, dim=1)
                loss = 0
                if num_ood > 0:
                    ets_outputs, ets_ood_outputs = ets_outputs.split((bs, num_ood), dim=0)
                    kbts_outputs, kbts_ood_outputs = kbts_outputs.split((bs, num_ood), dim=0)
                    
                    ets_ood_ent = entropy(ets_ood_outputs[:, :-1])
                    kbts_ood_ent = entropy(kbts_ood_outputs[:, :-1])

                    ets_ce_loss = F.nll_loss(ets_outputs.log(), labels, reduction='none') 
                    kbts_ce_loss = F.nll_loss(kbts_outputs.log(), labels, reduction='none')

                    loss += ets_ce_loss.mean() - self.alpha * ets_ood_ent.mean()
                    loss += kbts_ce_loss.mean() - self.alpha * kbts_ood_ent.mean()
                else:
                    ets_ce_loss = F.nll_loss(ets_outputs.log(), labels, reduction='none') 
                    kbts_ce_loss = F.nll_loss(kbts_outputs.log(), labels, reduction='none')
                    loss += ets_ce_loss.mean() + kbts_ce_loss.mean()

                if kd:
                    mask = (ets_ce_loss >= kbts_ce_loss).float()
                    loss += self.beta * torch.mean(mask * modified_kl_div(smooth(ets_outputs.detach(), 1, 1), smooth(kbts_outputs, 1, 1))
                                        + (1-mask) * modified_kl_div(smooth(kbts_outputs.detach(), 1, 1), smooth(ets_outputs, 1, 1)))

            
            assert not math.isnan(loss), f'{sum([m.kb_weight.norm(p=2) for m in self.net.DM[-1]])}'
            if self.args.amp:
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            else:
                loss.backward()
                self.opt.step()

            total += bs
            ets_total_loss += ets_ce_loss.mean().item() * bs
            kbts_total_loss += kbts_ce_loss.mean().item() * bs

            if not feat:
                # outputs = ensemble_outputs(torch.stack([ets_outputs, kbts_outputs], dim=1), dim=1)
                ets_correct += (ets_outputs.argmax(1) == labels).sum().item()
                kbts_correct += (kbts_outputs.argmax(1) == labels).sum().item()

            if squeeze and self.lamb[self.task] > 0:
                self.net.proximal_gradient_descent(self.scheduler.get_last_lr()[0], self.lamb[self.task])
            else:
                if 'wn' not in self.args.ablation:
                    self.net.normalize()
                
        if squeeze and self.lamb[self.task] > 0:
            self.net.squeeze(self.opt.state)
        self.scheduler.step()
        return ets_total_loss / total, round(ets_correct * 100 / total, 2), kbts_total_loss / total, round(kbts_correct * 100 / total, 2)
    
    def train_calibration(self):
        self.net.freeze(False)
        self.net.w_ets = torch.rand(self.task+1, requires_grad=True, device=self.args.device)
        self.net.b_ets = torch.rand(self.task+1, requires_grad=True, device=self.args.device)
        self.net.w_kbts = torch.rand(self.task+1, requires_grad=True, device=self.args.device)
        self.net.b_kbts = torch.rand(self.task+1, requires_grad=True, device=self.args.device)

        self.net.eval()
        optimizer = torch.optim.SGD([self.net.w, self.net.b], lr=0.01, momentum=0.8)
        total = 0
        ets_correct = 0
        ets_total_loss = 0

        kbts_correct = 0
        kbts_total_loss = 0
        for e in range(200):
            for i, data in enumerate(self.buffer):
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                ets_outputs = []
                kbts_outputs = []
                for t in range(self.task+1):
                    ets_outputs += [self.net.ets_forward(inputs, self.task, feat=False) * self.net.w_ets[t] + self.net.b_ets[t]]
                    kbts_outputs += [self.net.kbts_forward(inputs, self.task, feat=False) * self.net.w_kbts[t] + self.net.b_kbts[t]]
                
                ets_outputs = torch.cat(ets_outputs, dim=1)
                kbts_outputs = torch.cat(kbts_outputs, dim=1)
                loss = F.cross_entropy(ets_outputs, labels) + F.cross_entropy(kbts_outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += labels.shape[0]
                ets_correct += (ets_outputs.argmax(1) == labels).sum().item()
                kbts_correct += (kbts_outputs.argmax(1) == labels).sum().item()

            ets_acc = round(ets_correct * 100 / total, 2)
            kbts_acc = round(kbts_correct * 100 / total, 2)
            print(f'ets acc: {ets_acc}, kbts acc: {kbts_acc}')
            
    
    def back_updating(self, train_loader, t):
        total = 0
        correct = 0
        total_loss = 0
        # torch.cuda.empty_cache()
        
        self.net.train()
        # if self.buffer is not None:
        #     buffer = iter(self.buffer)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # if self.buffer is not None:
            #     try:
            #         buffer_data = next(buffer)
            #     except StopIteration:
            #         # restart the generator if the previous generator is exhausted.
            #         buffer = iter(self.buffer)
            #         buffer_data = next(buffer)
            #     inputs = torch.cat([inputs, buffer_data[0].to(self.device)], dim=0)
            # inputs = self.dataset.train_transform(inputs)

            self.opt.zero_grad()

            ets_outputs = self.net.ets_forward(inputs, t) 
            kbts_outputs = self.net.kbts_forward(inputs, t)

            loss = - entropy(F.softmax(ets_outputs, dim=1)).mean() - entropy(F.softmax(kbts_outputs, dim=1)).mean()
            
            assert not math.isnan(loss)
            loss.backward()
            self.net.proj_grad(t)
            self.opt.step()
            total += bs
            total_loss += loss.item() * bs

        self.scheduler.step()
        return total_loss / total, 0

    # def train_calibration(self, mode, ets, kbts):
        # self.net.train()
        # total = 0
        # correct = 0
        # total_loss = 0

        # for i, data in enumerate(self.buffer):
        #     self.opt.zero_grad()
        #     data = [tmp.to(self.device) for tmp in data]
        #     inputs = self.dataset.train_transform(data[0])

        #     outputs = []
        #     if ets:
        #         outputs += [torch.cat([self.net.ets_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) for t in range(self.task + 1)])]

        #     if kbts:
        #         outputs += [torch.cat([self.net.kbts_forward(self.dataset.test_transforms[t](inputs), t, feat=False, cal=True) for t in range(self.task + 1)])]

        #     outputs = torch.stack(outputs, dim=0)
        #     # outputs = outputs[:, :, 1:]  # ignore ood class
        #     outputs = ensemble_outputs(outputs)
        #     join_entropy = entropy(outputs.exp())
        #     join_entropy = join_entropy.view(self.task + 1, data[0].shape[0]).permute(1, 0)  # shape [batch size, num tasks]
        #     labels = torch.stack([(data[2] == t).float() for t in range(self.task + 1)], dim=1)
        #     loss = torch.sum(join_entropy * labels, dim=1) / (torch.sum(join_entropy, dim=1)+1e-9)
        #     # loss = - torch.sum(join_entropy * (1-labels), dim=1)
        #     loss = torch.mean(loss)

        #     assert not math.isnan(loss)
        #     loss.backward()
        #     self.opt.step()
        #     total += data[1].shape[0]
        #     total_loss += loss.item()

        # self.scheduler.step()
        # return total_loss / total, 0

    def begin_task(self, dataset):
        self.task += 1
        self.net.expand(dataset.N_CLASSES_PER_TASK, self.task)
        if 'init' not in self.args.ablation:
            self.net.initialize()
        self.net.ERK_sparsify(sparsity=self.args.sparsity)
        for m in self.net.DM:
            m.kbts_sparsities += [m.sparsity]
            # print(m.sparsity, m.strength)

    def end_task(self, dataset) -> None:
        # self.net.freeze_feature()
        # self.net.freeze_classifier()
        # self.net.check_var()
        self.net.freeze(False)
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
        
        # random class balanced selection
        indices = []
        data, targets = np.array(train_loader.dataset.data), np.array(train_loader.dataset.targets)
        for c in np.unique(targets):
            idx = np.arange(targets.shape[0])[targets == c][:samples_per_class]
            indices.append(idx)
        indices = np.concatenate(indices)
        data = np.array(data[indices])
        targets = np.array(targets[indices])

        if self.task > 0:
            buf_data, buf_targets = np.array(self.buffer.dataset.data), np.array(self.buffer.dataset.targets)
            data = np.concatenate([buf_data, data])
            targets = np.concatenate([buf_targets, targets])

        # self.buffer = DataLoader(Buffer(data, targets, self.dataset.TRANSFORM), batch_size=self.args.batch_size, shuffle=True)
        self.buffer = copy.deepcopy(train_loader)
        self.buffer.dataset.data = data
        self.buffer.dataset.targets = targets
        print('Buffer size', data.shape)
        # for c in np.unique(targets):
        #     print(sum(targets == c))

        # data = list(train_loader.dataset.tensors)
        # indices = []
        # for c in data[1].unique():
        #     idx = torch.arange(data[1].shape[0])[data[1] == c][:samples_per_class]
        #     indices.append(idx)
        # indices = torch.cat(indices)
        # data = [temp[indices] for temp in data]

        # if self.task > 0:
        #     buf_data = list(self.buffer.dataset.tensors)
        #     data = [torch.cat([buf_temp, temp]) for buf_temp, temp in zip(buf_data, data)]

        # self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        # print('Buffer size', data[0].shape)

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

        data, targets = np.array(self.buffer.dataset.data), np.array(self.buffer.dataset.targets)

        indices = []
        for c in np.unique(targets):
            idx = np.arange(targets.shape[0])[targets == c][:samples_per_class]
            indices.append(idx)
        indices = np.concatenate(indices)
        data = np.array(data[indices])
        targets = np.array(targets[indices])

        # self.buffer = DataLoader(Buffer(data, targets, self.dataset.TRANSFORM), batch_size=self.args.batch_size, shuffle=True)
        self.buffer.dataset.data = data
        self.buffer.dataset.targets = targets
        print('Buffer size', data.shape)
        # for c in np.unique(targets):
        #     print(sum(targets == c))

        # data = list(self.buffer.dataset.tensors)
        # indices = []
        # for c in data[1].unique():
        #     idx = torch.arange(data[1].shape[0])[data[1] == c][:samples_per_class]
        #     indices.append(idx)
        # indices = torch.cat(indices)
        # data = [temp[indices] for temp in data]

        # self.buffer = DataLoader(TensorDataset(*data), batch_size=self.args.batch_size, shuffle=True)
        # print('Buffer size', data[0].shape)

from PIL import Image
class Buffer():
    def __init__(self, data, targets, transform=None) -> None:
        self.transform = transform
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return self.targets.shape[0]


    def __getitem__(self, index):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target
