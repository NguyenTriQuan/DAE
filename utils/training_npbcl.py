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
# try:
#     import wandb
# except ImportError:
#     wandb = None

import wandb
def train_loop(t, model, dataset, args, progress_bar, train_loader, mode):
    squeeze = False
    augment = True
    num_squeeze = 0
    num_augment = 1000
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    
    model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=0, momentum=args.optim_mom)
    n_epochs = 120
    num_augment = 117
    model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [100, 115], gamma=0.1, verbose=False)

    if 'epoch' in args.ablation:
        n_epochs = 10
    for epoch in range(n_epochs):  
            
        loss, train_acc = model.train(train_loader, progress_bar, mode, squeeze, augment, epoch, args.verbose)
        if args.verbose:
            test_acc = model.evaluate([model.task], mode=mode)[0]
            dif = 0
            for m in model.net.DM:
                dif += (m.stable_masks[model.task] - m.plastic_masks[model.task]).abs().sum().int().item()   
            progress_bar.prog(epoch, n_epochs, epoch, model.task, loss, train_acc, test_acc, dif)
            wandb.log({f"task {model.task} loss": loss, f"task {model.task} train acc": train_acc, f"task {model.task} test acc": test_acc,  "epoch": epoch})
    print()


def evaluate(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # state_dict = torch.load(base_path_memory() + args.title + '.net')
    # model.net.load_state_dict(state_dict, strict=False)
    model.net = torch.load(base_path_memory() + args.title + '.net')
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        
        train_loader, test_loader = dataset.get_data_loaders()   
        model.task += 1 
        print(f'Task {t}:')
        model.net.update_unused_weights(t)        
        
        mode = 'ensemble'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        mode = 'stable'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        mode = 'plastic'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        if 'ba' not in args.ablation:
            # batch augmentation
            mode = 'ensemble_ba'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'stable_ba'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'plastic_ba'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

def train_cal(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    
    # state_dict = torch.load(base_path_memory() + args.title + '.net')
    # model.net.load_state_dict(state_dict, strict=False)
    model.net = torch.load(base_path_memory() + args.title + '.net')
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    model.net.ets_cal_layers = torch.nn.ModuleList([])
    model.net.kbts_cal_layers = torch.nn.ModuleList([])
    model.net.set_jr_params(args.num_tasks)
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        
        train_loader, test_loader = dataset.get_data_loaders()   
        model.task += 1 

        if 'kbts' not in args.ablation:
            eval_mode = 'ets_kbts'
        else:
            eval_mode = 'ets'

        eval_mode += '_cal'
        # model.net.set_jr_params(t+1)
        with torch.no_grad():
            model.get_rehearsal_logits(train_loader)
        # jr training
        if t > 0:
            # if 'tc' not in args.ablation:
            #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='tc')
                # if 'kbts' not in args.ablation:
                #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts_tc')

            train_loop(t, model, dataset, args, progress_bar, train_loader, mode='ets_cal')
            if 'kbts' not in args.ablation:
                train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts_cal')

            torch.save(model.net, base_path_memory() + args.title + '.net')

            til_accs = model.evaluate(task=range(t+1), mode=eval_mode)
            cil_accs = model.evaluate(task=None, mode=eval_mode)
            print(f'{eval_mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            eval_mode += '_ba'
            til_accs = model.evaluate(task=range(t+1), mode=eval_mode)
            cil_accs = model.evaluate(task=None, mode=eval_mode)
            print(f'{eval_mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'ets_cal'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'ets_cal_ba'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')


            mode = 'kbts_cal'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'kbts_cal_ba'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

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
    if 'sub' in args.ablation:
        ratio = 0.2
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

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        model.net.train()
        if 'joint' in args.ablation:
            train_loader, test_loader = dataset.get_full_data_loader()
        else:
            train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            print(f'Start training task {t}')
            model.begin_task(dataset)
        
        model.net.set_mode('ensemble')
        train_loop(t, model, dataset, args, progress_bar, train_loader, mode='ensemble')

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        torch.save(model.net, base_path_memory() + args.title + '.net')

        if args.verbose:
            mode = 'ensemble'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
            cil_avg = round(np.mean(cil_accs), 2)
            til_avg = round(np.mean(til_accs), 2)
            wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

            mode = 'stable'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
            cil_avg = round(np.mean(cil_accs), 2)
            til_avg = round(np.mean(til_accs), 2)
            wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

            mode = 'plastic'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
            cil_avg = round(np.mean(cil_accs), 2)
            til_avg = round(np.mean(til_accs), 2)
            wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

            if 'ba' not in args.ablation:
                # batch augmentation
                mode = 'ensemble_ba'
                til_accs = model.evaluate(task=range(t+1), mode=mode)
                cil_accs = model.evaluate(task=None, mode=mode)
                print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
                cil_avg = round(np.mean(cil_accs), 2)
                til_avg = round(np.mean(til_accs), 2)
                wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

                mode = 'stable_ba'
                til_accs = model.evaluate(task=range(t+1), mode=mode)
                cil_accs = model.evaluate(task=None, mode=mode)
                print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
                cil_avg = round(np.mean(cil_accs), 2)
                til_avg = round(np.mean(til_accs), 2)
                wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

                mode = 'plastic_ba'
                til_accs = model.evaluate(task=range(t+1), mode=mode)
                cil_accs = model.evaluate(task=None, mode=mode)
                print(f'Task {t}, mode {mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')
                cil_avg = round(np.mean(cil_accs), 2)
                til_avg = round(np.mean(til_accs), 2)
                wandb.log({f"{mode}_cil": cil_avg, f"{mode}_til": til_avg, "task": t})

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
