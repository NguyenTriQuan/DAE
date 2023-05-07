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

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, task=None, mode='ets_kbts_jr') -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    # status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if task is not None:
            if k != task:
                continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                # inputs = dataset.test_transform(inputs)
                if task is not None:
                    pred = model(inputs, k, mode)
                else:
                    pred = model(inputs, None, mode)

                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il' and task is None:
                    pred = model(inputs, k, mode)
                    correct_mask_classes += torch.sum(pred == labels).item()

        acc = correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0
        accs.append(round(acc, 2))
        acc = correct_mask_classes / total * 100
        accs_mask_classes.append(round(acc, 2))

    # model.net.train(status)
    return accs, accs_mask_classes

def train_loop(t, model, dataset, args, progress_bar, train_loader, mode):
    # model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=0, momentum=args.optim_mom)
    model.opt = torch.optim.SGD(model.net.get_optim_params(), lr=args.lr, weight_decay=0, momentum=args.optim_mom)
    squeeze = False
    num_squeeze = 70
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    if 'ets' in mode:
        lamb = model.lamb[t]
        print('lamb', lamb)
        if 'squeeze' in args.ablation:
            n_epochs = 10
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
            squeeze = False
        else:
            n_epochs = 100
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [85, 95], gamma=0.1, verbose=False)
            squeeze = True
        # n_epochs = 10
        # model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [85, 95], gamma=0.1, verbose=False)
        # squeeze = False
        if 'join' in args.ablation:
            n_epochs = 50
            model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
            squeeze = False
    elif 'kbts' in mode:
        n_epochs = 50
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
    elif 'jr' in mode:
        n_epochs = 50
        model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)

    for epoch in range(n_epochs):
        if mode == 'jr':
            model.train_rehearsal(progress_bar, epoch)
        else:          
            model.train(train_loader, progress_bar, mode, squeeze, epoch)

        if epoch >= num_squeeze:
            squeeze = False
    
    accs = evaluate(model, dataset, task=t, mode=mode)
    print('\n{} Accuracy for {} task(s): {} %'.format(mode, t+1, round(accs[0][0], 2)), file=sys.stderr)


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    args.title = '{}_{}_{}_{}_lamb_{}_drop_{}_sparsity_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset, 
                                                      args.ablation, args.lamb, args.dropout, args.sparsity)
    print(args.title)
    if args.debug:
        num = 1000
        dataset.train_data = dataset.train_data[:num]
        dataset.train_targets = dataset.train_targets[:num]
        dataset.test_data = dataset.test_data[:num]
        dataset.test_targets = dataset.test_targets[:num]
    model.dataset = dataset
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    args.ignore_other_metrics = True
    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy, ets=True, kbts=False, jr=False)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        if 'joint' in args.ablation:
            train_loader, test_loader = dataset.get_full_data_loader()
        else:
            train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
            num_params, num_neurons = model.net.count_params()
            print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')

        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True, ets=True, kbts=False, jr=False)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        
        # accs = evaluate(model, dataset, task=None, mode='ets')
        # mean_acc = np.mean(accs, axis=1)
        # print(f'init ets accs: cil {accs[0]}, til {accs[1]}')
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        # kbts training
        # train_loop(t, model, dataset, args, progress_bar, train_loader, mode='kbts')

        # ets training
        train_loop(t, model, dataset, args, progress_bar, train_loader, mode='ets')
        model.net.check_var()
        num_params, num_neurons = model.net.count_params()
        print(f'Num params :{sum(num_params)}, num neurons: {num_neurons}')

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset, task=None, mode='ets')
        mean_acc = np.mean(accs, axis=1)
        print(f'ets accs: cil {accs[0]}, til {accs[1]}')
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        # accs = evaluate(model, dataset, task=None, mode='ets_kbts')
        # mean_acc = np.mean(accs, axis=1)
        # print(f'ets_kbts accs: cil {accs[0]}, til {accs[1]}')
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        # with torch.no_grad():
        #     model.get_rehearsal_logits(train_loader)
        # jr training
        # train_loop(t, model, dataset, args, progress_bar, train_loader, mode='jr')

        # with torch.no_grad():
        #     model.fill_buffer(train_loader)

        # print('checking forgetting')
        # accs = evaluate(model, dataset, task=None, mode='kbts')
        # print(f'kbts accs: cil {accs[0]}, til {accs[1]}')

        # accs = evaluate(model, dataset, task=None, mode='ets')
        # print(f'ets accs: cil {accs[0]}, til {accs[1]}')

        # accs = evaluate(model, dataset, task=None, mode='jr')
        # print(f'jr accs: cil {accs[0]}, til {accs[1]}')

        # final evaluation
        # accs = evaluate(model, dataset, task=None, mode='ets_kbts_jr')
        # results.append(accs[0])
        # results_mask_classes.append(accs[1])
        # mean_acc = np.mean(accs, axis=1)
        # print(f'ets_kbts_jr accs: cil {accs[0]}, til {accs[1]}')
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        # save model and buffer
        model.net.clear_memory()
        torch.save(model, base_path_memory() + args.title + '.model')
        torch.save(dataset, base_path_memory() + args.title + '.dataset')
        torch.save(model.net.state_dict(), base_path_memory() + args.title + '.net')
        torch.save(model.buffers, base_path_memory() + args.title + '.buffer')
        # estimate memory size
        print('Model size:', os.path.getsize(base_path_memory() + args.title + '.net'))
        print('Buffer size:', os.path.getsize(base_path_memory() + args.title + '.buffer'))
        # print(model.net.state_dict().keys())
        # if not args.disable_log:
        #     logger.log(mean_acc)
        #     logger.log_fullacc(accs)

        # if not args.nowand:
        #     d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
        #         **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
        #         **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

        #     wandb.log(d2)



    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
