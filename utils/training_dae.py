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

wandb = None

# def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
#     """
#     Given the output tensor, the dataset at hand and the current task,
#     masks the former by setting the responses for the other tasks at -inf.
#     It is used to obtain the results for the task-il setting.
#     :param outputs: the output tensor
#     :param dataset: the continual dataset
#     :param k: the task index
#     """
#     outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
#     outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
#                dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


# def model.evaluate(model: ContinualModel, dataset: ContinualDataset, task=None, mode='ets_kbts_jr') -> Tuple[list, list]:
#     """
#     Evaluates the accuracy of the model for each past task.
#     :param model: the model to be evaluated
#     :param dataset: the continual dataset at hand
#     :return: a tuple of lists, containing the class-il
#              and task-il accuracy for each task
#     """
#     # status = model.net.training
#     with torch.no_grad():
#         model.net.eval()
#         accs, accs_mask_classes = [], []
#         for k, test_loader in enumerate(dataset.test_loaders):
#             if task is not None:
#                 if k != task:
#                     continue
#             correct, correct_mask_classes, total = 0.0, 0.0, 0.0
#             for data in test_loader:
#                 with torch.no_grad():
#                     inputs, labels = data
#                     inputs, labels = inputs.to(model.device), labels.to(model.device)
#                     # inputs = dataset.test_transform(inputs)
#                     if task is not None:
#                         pred = model(inputs, k, mode)
#                     else:
#                         pred = model(inputs, None, mode)

#                     correct += torch.sum(pred == labels).item()
#                     total += labels.shape[0]

#                     if dataset.SETTING == 'class-il' and task is None:
#                         pred = model(inputs, k, mode)
#                         correct_mask_classes += torch.sum(pred == labels).item()

#             acc = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
#             accs.append(round(acc, 2))
#             acc = correct_mask_classes / total * 100
#             accs_mask_classes.append(round(acc, 2))

#         # model.net.train(status)
#         return accs, accs_mask_classes

def train_loop(t, model, dataset, args, progress_bar, train_loader, mode):
    squeeze = False
    augment = True
    num_squeeze = 0
    num_augment = 100
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    if 'cal' in mode:
        # calibration outputs
        n_epochs = 100
        params = model.net.get_optim_cal_params()
        # params = list(model.net.ets_cal_head.parameters()) + list(model.net.ets_cal_layers.parameters()) \
        #             + list(model.net.kbts_cal_head.parameters()) + list(model.net.kbts_cal_layers.parameters())
        count = 0
        for param in params:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=args.optim_mom)
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [85, 95], gamma=0.1, verbose=False)
    elif 'tc' in mode:
        # tasks contrast:
        n_epochs = 200
        params = model.net.get_optim_tc_params()
        count = 0
        for param in params:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        from utils.lars_optimizer import LARC
        # model.opt = LARC(torch.optim.SGD(params, lr=args.lr, weight_decay=5e-3, momentum=0.9), trust_coefficient=0.001)
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=args.optim_mom)
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [180, 195], gamma=0.1, verbose=False)
        # model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.opt, T_max=n_epochs)
    elif 'ets' in mode:
        params = model.net.get_optim_ets_params()
        count = 0
        for param in params:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        model.opt = torch.optim.SGD(params, lr=args.lr, weight_decay=0, momentum=args.optim_mom)
        if 'squeeze' in args.ablation:
            n_epochs = 120
            num_augment = 100
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [100, 115], gamma=0.1, verbose=False)
            squeeze = False
        else:
            n_epochs = 150
            num_squeeze = 100
            num_augment = 130
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [130, 145], gamma=0.1, verbose=False)
            squeeze = True
        if 'join' in args.ablation:
            n_epochs = 50
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
            squeeze = False
    elif 'kbts' in mode:
        params, scores = model.net.get_optim_kbts_params()
        count = 0
        for param in params + scores:
            count += param.numel()
        print(f'Training mode: {mode}, Number of optim params: {count}')
        model.opt = torch.optim.SGD([{'params':params, 'lr':args.lr}, {'params':scores, 'lr':args.lr_score}], lr=args.lr, weight_decay=0, momentum=args.optim_mom)
        n_epochs = 120
        num_augment = 100
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [100, 115], gamma=0.1, verbose=False)

    if 'epoch' in args.ablation:
        n_epochs = 10
    for epoch in range(n_epochs):
        if 'cal' in mode:
            model.train_calibration(progress_bar, epoch, mode, args.verbose)
        elif 'tc' in mode:
            model.train_contrast(progress_bar, epoch, mode, args.verbose)
        else:          
            model.train(train_loader, progress_bar, mode, squeeze, augment, epoch, args.verbose)

        # do not perform squeeze and augment at a few last epochs for better convergence
        if epoch >= num_squeeze:
            squeeze = False

        if epoch >= num_augment:
            augment = False

    print()


def evaluate(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    # state_dict = torch.load(base_path_memory() + args.title + '.net')
    # model.net.load_state_dict(state_dict, strict=False)
    model.net = torch.load(base_path_memory() + args.title + '.net')
    num_params, num_neurons = model.net.count_params()
    num_neurons = '-'.join(str(int(num)) for num in num_neurons)
    print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
    for t in range(dataset.N_TASKS):
        if t >= args.num_tasks:
            break
        
        train_loader, test_loader = dataset.get_data_loaders()   
        model.task += 1 
        print(f'Task {t}:')
        num_params, num_neurons = model.net.count_params(t)
        num_neurons = '-'.join(str(int(num)) for num in num_neurons)
        print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
        
        mode = 'kbts'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        mode = 'ets'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        mode = 'ets_kbts'
        til_accs = model.evaluate(task=range(t+1), mode=mode)
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        mode = 'ets_kbts_ba'
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

        if 'cal' not in args.ablation:
            mode = 'ets_kbts_cal'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'ets_kbts_cal_ba'
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

            mode = 'ets_cal'
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

            mode = 'kbts_cal'
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

        mode = 'ets_ba'
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

        mode = 'kbts_ba'
        cil_accs = model.evaluate(task=None, mode=mode)
        print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}')

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
            if 'tc' not in args.ablation:
                train_loop(t, model, dataset, args, progress_bar, train_loader, mode='tc')
                # if 'kbts' not in args.ablation:
                #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts_tc')

            train_loop(t, model, dataset, args, progress_bar, train_loader, mode='ets_kbts_cal')
            # if 'kbts' not in args.ablation:
            #     train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts_cal')

            til_accs = model.evaluate(task=range(t+1), mode=eval_mode)
            cil_accs = model.evaluate(task=None, mode=eval_mode)
            print(f'{eval_mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            eval_mode += '_ba'
            til_accs = model.evaluate(task=range(t+1), mode=eval_mode)
            cil_accs = model.evaluate(task=None, mode=eval_mode)
            print(f'{eval_mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

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
        
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

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
            model.begin_task(dataset)
            num_params, num_neurons = model.net.count_params()
            num_neurons = '-'.join(str(int(num)) for num in num_neurons)
            print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')
            if t == 0:
                base_params = sum(num_params)
                model.factor = 1
            else:
                model.factor = sum(num_params) / base_params
            print(f'Task {t}, lamb = {model.lamb * model.factor}')

        # kbts training
        if 'kbts' not in args.ablation:
            mode = 'kbts'
            train_loop(t, model, dataset, args, progress_bar, train_loader, mode=mode)
            if args.verbose:
                accs = model.evaluate(task=[t], mode=mode)
                print(f'Task {t}, {mode}: til {accs[0]}')

        model.net.clear_memory()

        # ets training
        if 'ets' not in args.ablation:
            mode = 'ets'
            train_loop(t, model, dataset, args, progress_bar, train_loader, mode=mode)
            if args.verbose:
                accs = model.evaluate(task=[t], mode=mode)
                print(f'Task {t}, {mode}: til {accs[0]}')
                num_params, num_neurons = model.net.count_params()
                num_neurons = '-'.join(str(int(num)) for num in num_neurons)
                print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        if 'kbts' not in args.ablation:
            eval_mode = 'ets_kbts'
        else:
            eval_mode = 'ets'

        if args.verbose:
            cil_accs = model.evaluate(task=None, mode=eval_mode)
            til_accs = model.evaluate(task=[t], mode=eval_mode)
            print(f'Task {t}, {eval_mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {til_accs[0]}')

            if 'ba' not in args.ablation:
                # batch augmentation
                accs = model.evaluate(task=None, mode=eval_mode+'_ba')
                print(f'Task {t}, {eval_mode}_ba: cil {round(np.mean(accs), 2)} {accs}')

        if args.verbose:
            print('checking forgetting')
            mode = 'kbts'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

            mode = 'ets'
            til_accs = model.evaluate(task=range(t+1), mode=mode)
            cil_accs = model.evaluate(task=None, mode=mode)
            print(f'{mode}: cil {round(np.mean(cil_accs), 2)} {cil_accs}, til {round(np.mean(til_accs), 2)} {til_accs}')

        # torch.save(model.net.state_dict(), base_path_memory() + args.title + '.net')
        torch.save(model.net, base_path_memory() + args.title + '.net')
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
