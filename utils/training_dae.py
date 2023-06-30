# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar
from utils.conf import base_path_memory
from utils.lars_optimizer import LARC
# try:
#     import wandb
# except ImportError:
#     wandb = None

import wandb

def train_loop(model, args, train_loader, mode, checkpoint=None, t=0):
    start_epoch = 0
    if checkpoint is not None:
        start_epoch = checkpoint['epoch']+1
        mode = checkpoint['mode']

    squeeze = False
    augment = True
    ets = 'ets' in mode
    kbts = 'kbts' in mode
    buf = 'buf' in args.mode
    rot = 'rot' in args.mode
    adv = 'adv' in args.mode
    feat = 'feat' in mode
    head = 'head' in mode
    cal = 'cal' in mode
    # all = 'all' in mode
    # if all: feat = False
    num_squeeze = 0

    if cal:
        model.net.freeze_feature()
        model.net.last.weight_ets[t].requires_grad = True
        model.net.last.weight_kbts[t].requires_grad = True
        params = [model.net.last.weight_ets[t], model.net.last.weight_kbts[t]]
        n_epochs = 50
        num_squeeze = 0
        step_lr = [35, 45]
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)
        count = 0
        feat = False
        for param in params:
            count += param.numel()
    elif feat:
        ets_params = model.net.get_optim_ets_params()
        kbts_params, scores = model.net.get_optim_kbts_params()
        n_epochs = 200
        num_squeeze = 100
        step_lr = [160, 190]
        squeeze = 'squeeze' not in args.ablation
        model.opt = torch.optim.SGD([{'params':ets_params+kbts_params, 'lr':args.lr}, {'params':scores, 'lr':args.lr_score}], 
                                    lr=args.lr, weight_decay=0, momentum=0.9)
        count = 0
        for param in ets_params+kbts_params+scores:
            count += param.numel()
    elif head:
        model.net.freeze_feature()
        params = model.net.last.get_optim_ets_params() + model.net.last.get_optim_kbts_params()
        n_epochs = 50
        num_squeeze = 0
        step_lr = [35, 45]
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)
        count = 0
        feat = False
        for param in params:
            count += param.numel()
    else:
        ets_params = model.net.get_optim_ets_params()
        kbts_params, scores = model.net.get_optim_kbts_params()
        n_epochs = 150
        num_squeeze = 100
        step_lr = [130, 145]
        squeeze = 'squeeze' not in args.ablation
        model.opt = torch.optim.SGD([{'params':ets_params+kbts_params, 'lr':args.lr}, {'params':scores, 'lr':args.lr_score}], 
                                    lr=args.lr, weight_decay=0, momentum=0.9)
        count = 0
        for param in ets_params+kbts_params+scores:
            count += param.numel()

    print(f'Training mode: {mode}, Number of optim params: {count}')
    model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, step_lr, gamma=0.1, verbose=False)

    if checkpoint is not None:
        model.opt = checkpoint['opt']
        model.scheduler = checkpoint['scheduler']

    if 'epoch' in args.ablation:
        n_epochs = 10
    progress_bar = ProgressBar()
    for epoch in range(start_epoch, n_epochs):
        if cal:
            loss = model.back_updating(train_loader, t)
        else:          
            loss, train_acc = model.train_contrast(train_loader, mode, ets, kbts, rot, buf, adv, feat, squeeze, augment)

        # wandb.save(base_path_memory() + args.title + '.tar')
        if args.verbose:
            test_acc = 0
            if not feat and not cal:
                test_acc = model.evaluate(task=model.task, mode=mode)
            if squeeze:
                num_params, num_neurons = model.net.count_params()
                num_neurons = '-'.join(str(int(num)) for num in num_neurons)
                num_params = sum(num_params)
                progress_bar.prog(epoch, n_epochs, epoch, model.task, loss, train_acc, test_acc, num_params, num_neurons)
                wandb.log({'epoch':epoch, f"Task {model.task} {mode} loss": loss, f"Task {model.task} {mode} train acc": train_acc,
                           f"Task {model.task} {mode} test acc": test_acc, f"Task {model.task} params": num_params})
            else:
                progress_bar.prog(epoch, n_epochs, epoch, model.task, loss, train_acc, test_acc)
                wandb.log({'epoch':epoch, f"Task {model.task} {mode} loss": loss, f"Task {model.task} {mode} train acc": train_acc,
                           f"Task {model.task} {mode} test acc": test_acc})

        #save model
        model.net.clear_memory()
        checkpoint = {'net': model.net, 'opt': model.opt, 'scheduler': model.scheduler, 'epoch':epoch, 'mode':mode, 'task':model.task}
        torch.save(checkpoint, base_path_memory() + args.title + '.checkpoint')

        if epoch >= num_squeeze:
            squeeze = False

    print()
    if ets:
        num_params, num_neurons = model.net.count_params()
        num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')

    # wandb.save(base_path_memory() + args.title + '.checkpoint')


def evaluate(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # state_dict = torch.load(base_path_memory() + args.title + '.net')
    # model.net.load_state_dict(state_dict, strict=False)
    # model.net = torch.load(base_path_memory() + args.title + '.net')
    # artifact = args.run.use_artifact('entity/DAE/model:v0', type='model')
    # artifact_dir = artifact.download()
    # model.net = torch.load(artifact_dir)
    # if wandb.run.resumed:
    checkpoint = torch.load(base_path_memory() + args.title + '.checkpoint')
    model.net = checkpoint['net']

    num_params, num_neurons = model.net.count_params()
    num_neurons = '-'.join(str(int(num)) for num in num_neurons)
    print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        
        model.task += 1
        train_loader, test_loader = dataset.get_data_loaders()   
        print(f'Task {t}:')
        num_params, num_neurons = model.net.count_params(t)
        num_params = sum(num_params)
        num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        print(f'Num params :{num_params}, num neurons: {num_neurons}')
        if args.verbose:
            wandb.log({'params': num_params, 'task': t})

        if args.task >= 0 :
            if t != args.task:
                continue
        
        mode = 'ets_kbts'
        model.evaluate(task=None, mode=mode)

        mode = 'ets_kbts_ba'
        model.evaluate(task=None, mode=mode)

        # if 'cal' in args.mode:
        #     mode = 'ets_kbts_cal'
        #     model.evaluate(task=None, mode=mode)

        #     mode = 'ets_kbts_ba_cal'
        #     model.evaluate(task=None, mode=mode)

        mode = 'ets'
        model.evaluate(task=None, mode=mode)

        mode = 'ets_ba'
        model.evaluate(task=None, mode=mode)

        # if 'cal' in args.mode:
        #     mode = 'ets_cal'
        #     model.evaluate(task=None, mode=mode)

        #     mode = 'ets_ba_cal'
        #     model.evaluate(task=None, mode=mode)

        mode = 'kbts'
        model.evaluate(task=None, mode=mode)

        mode = 'kbts_ba'
        model.evaluate(task=None, mode=mode)

        # if 'cal' in args.mode:
        #     mode = 'kbts_cal'
        #     model.evaluate(task=None, mode=mode)

        #     mode = 'kbts_ba_cal'
        #     model.evaluate(task=None, mode=mode)

def train_cal(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
   
    print(args)
    start_task = 0
    checkpoint = torch.load(base_path_memory() + args.title + '.checkpoint')
    start_task = checkpoint['task']
    model.net = checkpoint['net']
    model.net.to(model.device)
    print(model.net.kbts_sparsities)

    print(file=sys.stderr)

    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks: break
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        model.task += 1
        print(f'Training task {model.task}')
        model.net.get_representation_matrix(train_loader, t)
        for i in range(t):
            train_loop(model, args, train_loader, mode='ets_kbts_cal', t=i)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if args.verbose:
            mode = []
            if 'kbts' not in args.ablation:
                mode += ['kbts']
            if 'ets' not in args.ablation:
                mode += ['ets']
            mode = '_'.join(mode)
            model.evaluate(task=None, mode=mode)

            if 'ba' not in args.ablation:
                mode += '_ba'
                model.evaluate(task=None, mode=mode)
            
            if 'ets' not in args.ablation:
                mode = 'ets'
                model.evaluate(task=None, mode=mode)

            # mode = 'ets_ba'
            # model.evaluate(task=None, mode=mode)
            if 'kbts' not in args.ablation:
                mode = 'kbts'
                model.evaluate(task=None, mode=mode)

            # mode = 'kbts_ba'
            # model.evaluate(task=None, mode=mode)


        with torch.no_grad():
            model.get_rehearsal_logits(train_loader)

        with torch.no_grad():
            model.fill_buffer(train_loader)


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    start_task = 0
    checkpoint = None
    if args.resume:
        try:
            checkpoint = torch.load(base_path_memory() + args.title + '.checkpoint')
            # checkpoint = torch.load(wandb.restore('tmp/memory/' + args.title + '.checkpoint'))
            start_task = checkpoint['task']
            model.net = checkpoint['net']
        except Exception as e:
            print(e)
            print('no checkpoint to resume')

    if 'sub' in args.ablation:
        ratio = 0.1
        data = []
        targets = []
        for c in dataset.train_targets.unique():
            idx = dataset.train_targets == c
            num = int(ratio * idx.sum())
            data.append(dataset.train_data[idx][:num])
            targets.append(dataset.train_targets[idx][:num])
        dataset.train_data = torch.cat(data)
        dataset.train_targets = torch.cat(targets)
        print(dataset.train_data.shape)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    # if not args.disable_log:
        # logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    print(file=sys.stderr)

    
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks: break
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task') and checkpoint is None:
            model.begin_task(dataset)
            model.net.to(model.device)
            num_params, num_neurons = model.net.count_params()
            num_neurons = '-'.join(str(int(num)) for num in num_neurons)
            print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
        else:
            model.task += 1

        print(f'Training task {model.task}')
        if t >= start_task:
            # modes = ['ets', 'kbts']
            if 'feat' in args.mode:
                modes = ['ets_kbts_feat', 'ets_kbts_head']
            else:
                modes = ['ets_kbts']
            if checkpoint is not None:
                for mode in modes:
                    if mode == checkpoint['mode']:
                        break
                    else:
                        modes.remove(mode)

            for mode in modes:
                train_loop(model, args, train_loader, mode=mode, checkpoint=checkpoint)
                acc = model.evaluate(task=t, mode=mode)
                print(f'Task {t}, {mode}: til {acc}')
                checkpoint = None

            if hasattr(model, 'end_task'):
                model.end_task(dataset)

        if args.verbose and t >= start_task:
            mode = []
            if 'kbts' not in args.ablation:
                mode += ['kbts']
            if 'ets' not in args.ablation:
                mode += ['ets']
            mode = '_'.join(mode)
            model.evaluate(task=None, mode=mode)

            if 'ba' not in args.ablation:
                mode += '_ba'
                model.evaluate(task=None, mode=mode)
            
            if 'ets' not in args.ablation:
                mode = 'ets'
                model.evaluate(task=None, mode=mode)

            # mode = 'ets_ba'
            # model.evaluate(task=None, mode=mode)
            if 'kbts' not in args.ablation:
                mode = 'kbts'
                model.evaluate(task=None, mode=mode)

            # mode = 'kbts_ba'
            # model.evaluate(task=None, mode=mode)


        with torch.no_grad():
            model.get_rehearsal_logits(train_loader)


        with torch.no_grad():
            model.fill_buffer(train_loader)

