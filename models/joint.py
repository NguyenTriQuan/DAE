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
from utils.status import progress_bar


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)

    def forward(self, inputs, ba=True):
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
                    wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, f"{mode}_tp": task_acc, "task": len(til_accs) - 1})
                return til_accs, cil_accs, task_acc
            else:
                return til_accs[0]

    def train(self, train_loader, mode, ets, kbts, clr_ood, buf_ood, feat, squeeze, augment):
        total = 0
        correct = 0
        total_loss = 0

        self.net.train()

        if self.buffer is not None:
            buffer = iter(self.buffer)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            bs = labels.shape[0]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels - self.task * self.dataset.N_CLASSES_PER_TASK
            if clr_ood:
                rot = random.randint(1, 3)
                ood_inputs = torch.rot90(inputs, rot, dims=(2, 3))
                if buf_ood:
                    if self.buffer is not None:
                        try:
                            buffer_data = next(buffer)
                        except StopIteration:
                            # restart the generator if the previous generator is exhausted.
                            buffer = iter(self.buffer)
                            buffer_data = next(buffer)
                        buffer_data = [tmp.to(self.device) for tmp in buffer_data]
                        ood_inputs = torch.cat([ood_inputs, buffer_data[0]], dim=0)

            if feat:
                inputs = torch.cat([inputs, inputs], dim=0)
                if clr_ood:
                    inputs = torch.cat([inputs, ood_inputs], dim=0)
            else:
                if clr_ood:
                    # ood_labels = torch.zeros(ood_inputs.shape[0], dtype=torch.long).to(device)
                    inputs = torch.cat([inputs, ood_inputs], dim=0)
                    # labels = torch.cat([labels, ood_labels], dim=0)
            if augment:
                inputs = self.dataset.train_transform(inputs)
            inputs = self.dataset.test_transforms[self.task](inputs)
            self.opt.zero_grad()
            if ets:
                outputs = self.net.ets_forward(inputs, self.task, feat=feat)
            elif kbts:
                outputs = self.net.kbts_forward(inputs, self.task, feat=feat)

            if feat:
                outputs = F.normalize(outputs, p=2, dim=1)
                if clr_ood:
                    ind_outputs = outputs[:bs*2]
                    loss = sup_clr_ood_loss(ind_outputs, outputs, labels, self.args.temperature)
                else:
                    loss = sup_clr_loss(outputs, labels, self.args.temperature)
            else:
                if clr_ood:
                    ind_outputs = outputs[:bs]
                    ood_outputs = outputs[bs:]
                    # loss = (self.loss(ind_outputs, labels) + self.loss(ood_outputs, ood_labels)) / 2
                    ood_outputs = ensemble_outputs(ood_outputs.unsqueeze(0))
                    loss = self.loss(ind_outputs, labels) - self.alpha * entropy(ood_outputs.exp()).mean()
                else:
                    loss = self.loss(outputs, labels)
            assert not math.isnan(loss)
            loss.backward()
            self.opt.step()
            total += bs
            total_loss += loss.item() * bs
            if squeeze:
                self.net.proximal_gradient_descent(self.scheduler.get_last_lr()[0], self.lamb[self.task])
                
        if squeeze:
            self.net.squeeze(self.opt.state)
        self.scheduler.step()

        return total_loss / total
    
    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            self.old_data.append(dataset.train_loader)
            # train
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            all_inputs = []
            all_labels = []
            for source in self.old_data:
                for x, l, _ in source:
                    all_inputs.append(x)
                    all_labels.append(l)
            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)
            bs = self.args.batch_size
            scheduler = dataset.get_scheduler(self, self.args)

            for e in range(self.args.n_epochs):
                order = torch.randperm(len(all_inputs))
                for i in range(int(math.ceil(len(all_inputs) / bs))):
                    inputs = all_inputs[order][i * bs: (i + 1) * bs]
                    labels = all_labels[order][i * bs: (i + 1) * bs]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

                if scheduler is not None:
                    scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
