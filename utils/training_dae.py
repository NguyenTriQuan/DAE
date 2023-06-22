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

def train_loop(model, args, train_loader, mode, checkpoint=None):
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
    cal = 'cal' in mode
    # all = 'all' in mode
    # if all: feat = False
    num_squeeze = 0
    num_augment = 1000

    if cal:
        # calibration outputs
        n_epochs = 50
        tc = 'tc' not in args.ablation
        params = model.net.get_optim_cal_params(tc)
        count = 0
        for param in params:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
    # elif 'tc' in mode:
    #     # tasks contrast:
    #     n_epochs = 100
    #     params = model.net.get_optim_tc_params()
    #     count = 0
    #     for param in params:
    #         count += param.numel()
    #     print(f'Training mode: {mode}, Number of optim params: {count}')
    #     from utils.lars_optimizer import LARC
    #     # model.opt = LARC(torch.optim.SGD(params, lr=args.lr, weight_decay=5e-3, momentum=0.9), trust_coefficient=0.001)
    #     model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=args.optim_mom)
    #     model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [85, 95], gamma=0.1, verbose=False)
    #     # model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.opt, T_max=n_epochs)
    elif ets:
        # if feat or all:
        params = model.net.get_optim_ets_params()
        n_epochs = 150
        num_squeeze = 100
        step_lr = [130, 145]
        squeeze = 'squeeze' not in args.ablation
        # model.opt = LARC(torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9), trust_coefficient=0.001)
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)
        # else:
        #     params = model.net.last.get_optim_ets_params()
        #     n_epochs = 50
        #     step_lr = [1, 35, 45]
        #     model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)
        
        count = 0
        for param in params:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, step_lr, gamma=0.1, verbose=False)
        
    elif kbts:
        # if feat or all:
        n_epochs = 120
        step_lr = [100, 115]
        params, scores = model.net.get_optim_kbts_params()
        count = 0
        for param in params + scores:
            count += param.numel()
        model.opt = torch.optim.SGD([{'params':params, 'lr':args.lr}, {'params':scores, 'lr':args.lr_score}], 
                                    lr=args.lr, weight_decay=0, momentum=0.9)
        # model.opt = LARC(torch.optim.SGD([{'params':params, 'lr':args.lr}, {'params':scores, 'lr':args.lr_score}], 
        #                                  lr=args.lr, weight_decay=0, momentum=0.9), trust_coefficient=0.001)
        # else:
        #     n_epochs = 50
        #     step_lr = [1, 35, 45]
        #     params = model.net.last.get_optim_kbts_params()
        #     count = 0
        #     for param in params:
        #         count += param.numel()
        #     model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=0.9)

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
            loss = model.train_calibration(mode, ets, kbts)
        else:          
            loss, train_acc = model.train_contrast(train_loader, mode, ets, kbts, rot, buf, adv, squeeze, augment)

        # wandb.save(base_path_memory() + args.title + '.tar')
        if args.verbose:
            test_acc = 0
            # if not feat:
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
    checkpoint = torch.load(base_path_memory() + args.title + '.tar')
    model.net = checkpoint['net']

    num_params, num_neurons = model.net.count_params()
    num_neurons = '-'.join(str(int(num)) for num in num_neurons)
    print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        if args.task >= 0 :
            if t != args.task:
                continue
        model.task += 1
        train_loader, test_loader = dataset.get_data_loaders()   
        print(f'Task {t}:')
        num_params, num_neurons = model.net.count_params(t)
        num_params = sum(num_params)
        num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        print(f'Num params :{num_params}, num neurons: {num_neurons}')
        if args.verbose:
            wandb.log({'params': num_params, 'task': t})
        
        mode = 'ets_kbts'
        model.evaluate(task=None, mode=mode)

        mode = 'ets_kbts_ba'
        model.evaluate(task=None, mode=mode)

        if 'cal' in args.mode:
            mode = 'ets_kbts_cal'
            model.evaluate(task=None, mode=mode)

            mode = 'ets_kbts_ba_cal'
            model.evaluate(task=None, mode=mode)

        mode = 'ets'
        model.evaluate(task=None, mode=mode)

        mode = 'ets_ba'
        model.evaluate(task=None, mode=mode)

        if 'cal' in args.mode:
            mode = 'ets_cal'
            model.evaluate(task=None, mode=mode)

            mode = 'ets_ba_cal'
            model.evaluate(task=None, mode=mode)

        mode = 'kbts'
        model.evaluate(task=None, mode=mode)

        mode = 'kbts_ba'
        model.evaluate(task=None, mode=mode)

        if 'cal' in args.mode:
            mode = 'kbts_cal'
            model.evaluate(task=None, mode=mode)

            mode = 'kbts_ba_cal'
            model.evaluate(task=None, mode=mode)

def train_cal(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    
    # state_dict = torch.load(base_path_memory() + args.title + '.net')
    # model.net.load_state_dict(state_dict, strict=False)
    # model.net = torch.load(base_path_memory() + args.title + '.net')
    checkpoint = torch.load(base_path_memory() + args.title + '.tar')
    model.net = checkpoint['net']
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    model.net.set_cal_params(args.total_tasks)
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        model.task += 1
        model.net.set_cal_params(args.total_tasks)
        train_loader, test_loader = dataset.get_data_loaders()   
        with torch.no_grad():
            model.get_rehearsal_logits(train_loader)
        print('Task', model.task)
        if 'kbts' not in args.ablation:
            eval_mode = 'ets_kbts'
        else:
            eval_mode = 'ets'

        eval_mode += '_cal'

        run  = t > 0
        if args.task >= 0 :
            if t != args.task:
                run = False
        if run:
            # if 'tc' not in args.ablation:
            #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='tc')
                # if 'kbts' not in args.ablation:
                #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts_tc')

            train_loop(model, args, train_loader, mode='ets_cal')
            if 'kbts' not in args.ablation:
                train_loop(model, args, train_loader, mode='kbts_cal')

            model.evaluate(task=None, mode=eval_mode)

            eval_mode += '_ba'
            model.evaluate(task=None, mode=eval_mode)

            mode = 'ets'
            model.evaluate(task=None, mode=mode)

            mode = 'ets_cal'
            model.evaluate(task=None, mode=mode)

            mode = 'ets_cal_ba'
            model.evaluate(task=None, mode=mode)

            mode = 'kbts'
            model.evaluate(task=None, mode=mode)

            mode = 'kbts_cal'
            model.evaluate(task=None, mode=mode)

            mode = 'kbts_cal_ba'
            model.evaluate(task=None, mode=mode)

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
    if wandb.run.resumed and args.resume:
        checkpoint = torch.load(base_path_memory() + args.title + '.checkpoint')
        # checkpoint = torch.load(wandb.restore('tmp/memory/' + args.title + '.checkpoint'))
        start_task = checkpoint['task']
        model.net = checkpoint['net']

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

    if 'cal' in args.mode:
        model.net.set_cal_params(args.total_tasks)
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
            modes = ['ets', 'kbts']
            if checkpoint is not None:
                for mode in modes:
                    if mode == checkpoint['mode']:
                        break
                    else:
                        modes.remove(mode)

            for mode in modes:
                if 'ets' in mode and 'ets' in args.ablation:
                    continue
                if 'kbts' in mode and 'kbts' in args.ablation:
                    continue
                train_loop(model, args, train_loader, mode=mode, checkpoint=checkpoint)
                acc = model.evaluate(task=t, mode=mode)
                print(f'Task {t}, {mode}: til {acc}')
                checkpoint = None

            if hasattr(model, 'end_task'):
                model.end_task(dataset)
        # else:
        #     # ets training
        #     mode = 'ets_feat'
        #     train_loop(model, args, train_loader, mode=mode)
        #     num_params, num_neurons = model.net.count_params()
        #     num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        #     print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')

        #     # kbts training
        #     if 'kbts' not in args.ablation:
        #         mode = 'kbts_feat'
        #         train_loop(model, args, train_loader, mode=mode)
                
        #     model.net.freeze_feature()
        #     model.net.clear_memory()

        #     mode = 'ets'
        #     train_loop(model, args, train_loader, mode=mode)
        #     acc = model.evaluate(task=t, mode=mode)
        #     print(f'Task {t}, {mode}: til {acc}')

        #     if 'kbts' not in args.ablation:
        #         mode = 'kbts'
        #         train_loop(model, args, train_loader, mode=mode)
        #         acc = model.evaluate(task=t, mode=mode)
        #         print(f'Task {t}, {mode}: til {acc}')

        # model.net.clear_memory()
        # torch.save(model.net, base_path_memory() + args.title + '.net')
        # model_size = os.path.getsize(base_path_memory() + args.title + '.net')
        # print('Model size:', model_size)
        # if args.verbose:
        #     wandb.log({'model size':model_size, 'task': t})

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

        # if 'cal' in args.mode:
        #     model.net.set_cal_params(args.total_tasks)
        #     if t > 0:
        #         if 'ets' in args.mode:
        #             train_loop(model, args, train_loader, mode='ets_cal')
        #         if 'kbts' in args.mode:
        #             train_loop(model, args, train_loader, mode='kbts_cal')

        #         if args.verbose:
        #             mode = 'ets_kbts_cal'
        #             model.evaluate(task=None, mode=mode)

        #             mode = 'ets_kbts_cal_ba'
        #             model.evaluate(task=None, mode=mode)

        #             mode = 'ets_cal'
        #             model.evaluate(task=None, mode=mode)

        #             mode = 'ets_cal_ba'
        #             model.evaluate(task=None, mode=mode)

        #             mode = 'kbts_cal'
        #             model.evaluate(task=None, mode=mode)

        #             mode = 'kbts_cal_ba'
        #             model.evaluate(task=None, mode=mode)

        with torch.no_grad():
            model.fill_buffer(train_loader)


        # torch.save(model.net.state_dict(), base_path_memory() + args.title + '.net')
        # torch.save(model.buffers, base_path_memory() + args.title + '.buffer')
        # estimate memory size
        # print('Model size:', os.path.getsize(base_path_memory() + args.title + '.net'))
        # print(model.net.state_dict().keys())
        # print('Buffer size:', os.path.getsize(base_path_memory() + args.title + '.buffer'))
        # print(model.net.state_dict().keys())
        # if not args.disable_log:
        #     logger.log(mean_acc)
        #     logger.log_fullacc(accs)

        # if not args.nowand:
        #     d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
        #         **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
        #         **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

        #     wandb.log(d2)



    # if not args.disable_log and not args.ignore_other_metrics:
    #     logger.add_bwt(results, results_mask_classes)
    #     logger.add_forgetting(results, results_mask_classes)
    #     if model.NAME != 'icarl' and model.NAME != 'pnn':
    #         logger.add_fwt(results, random_results_class,
    #                 results_mask_classes, random_results_task)

    # if not args.disable_log:
    #     logger.write(vars(args))
    #     if not args.nowand:
    #         d = logger.dump()
    #         d['wandb_url'] = wandb.run.get_url()
    #         wandb.log(d)

    # if not args.nowand:
    #     wandb.finish()
