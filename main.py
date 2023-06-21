# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys


# mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(mammoth_path)
# sys.path.append(mammoth_path + '/datasets')
# sys.path.append(mammoth_path + '/backbone')
# sys.path.append(mammoth_path + '/models')

import datetime
import uuid
from argparse import ArgumentParser

# import setproctitle
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
import inspect

from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train
import wandb

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    # lecun_fix()
    if args is None:
        args = parse_args()

    # os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    # os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)
    dataset.N_TASKS = args.total_tasks
    dataset.N_CLASSES_PER_TASK = dataset.N_CLASSES // args.total_tasks

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset)

    if args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        args.nowand = 1

    # args.nowand = True
    # args.disable_log = True
    # set job name
    # setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    if model.NAME == 'DAE':
        from utils.training_dae import train, evaluate, train_cal
        args.title = '{}_{}_{}_lamb_{}_drop_{}_sparsity_{}_buf_{}_{}'.format(args.model, args.dataset, args.total_tasks,
                                                                args.lamb, args.dropout, args.sparsity,
                                                                 args.buffer_size, args.mode)
        print(args.title)
    elif model.NAME == 'ATA':
        from utils.training_ata import train
        args.title = '{}_{}_lamb_{}_drop_{}_sparsity_{}'.format(args.model, args.dataset, 
                                                                args.lamb, args.dropout, args.sparsity)
        print(args.title)
    elif model.NAME == 'NPBCL':
        from utils.training_npbcl import train, evaluate, train_cal
        args.title = '{}_{}_lamb_{}_alpha_{}_beta_{}_sparsity_{}'.format(args.model, args.dataset, 
                                                                args.lamb, args.alpha, args.beta, args.sparsity)
        print(args.title)
    else:
        from utils.training import train
        args.title = '{}_{}_batch_{}_lr_{}'.format(args.model, args.dataset, 
                                                                args.batch_size, args.lr)
        print(args.title)

    if args.verbose:
        wandb.login(key='74ac7eba00fea7e805a70861a86c7767406946c9')
        run = wandb.init(
            # Set the project where this run will be logged
            project=model.NAME,
            name=args.title,
            resume=True,
            # Track hyperparameters and run metadata
            config=args
            # config={
            #     'dataset': args.dataset,
            #     'total tasks': args.total_tasks,
            #     "learning rate": args.lr,
            #     "learning score": args.lr_score,
            #     'lamb': args.lamb,
            #     'sparsity': args.sparsity,
            #     'dropout': args.dropout,
            #     'buffer': args.buffer_size,
            #     'ablation': args.ablation,
            #     'temperature': args.temperature
            # }
            )
    if args.eval:
        evaluate(model, dataset, args)
    elif args.cal:
        train_cal(model, dataset, args)
    elif model.NAME == 'joint':
        model.train_loop()
    elif isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)
    if args.verbose:
        wandb.finish()


if __name__ == '__main__':
    main()
