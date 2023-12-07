#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19
import torch.optim as optim
from torch.utils.data import DataLoader
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from deepspeed.accelerator import get_accelerator

# define by myself
from sampler import Distributed_Elastic_Sampler
from typing import Iterable, TypeVar
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--epochs',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed',
                        type=int,
                        default=1138,
                        help='PRNG seed')
    parser.add_argument('--log_interval',
                        type=int,
                        default=300,
                        help='print every log_interval mini-batches losses')
    parser.add_argument('--data_partition_method',
                        type=str,
                        default='manual',
                        help='A method about how to split data between different processes.')
    parser.add_argument('--manual_partition_list',
                        type=list,
                        default=[30000, 20000],
                        help='different processes\' data proportion')

    parser.add_argument('--batch_size',
                        type=int,
                        default=[30,20],
                        help='In accordance with deepspeed json file:train_micro_batch_size_per_gpu')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


T = TypeVar("T")


class InfiniteIterator(object):
    def __init__(self, iterable: Iterable[T]) -> None:
        self._iterable = iterable
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self) -> T:
        next_item = None
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item


def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset


def dataset_split(
        args,
        dataset: torch.utils.data.Dataset
) -> InfiniteIterator:

    sampler_dict = \
        {
            'method': args.data_partition_method,
            'manual_partition_list': args.manual_partition_list
        }

    sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size[args.local_rank],
                              shuffle=False,
                              sampler=sampler,
                              pin_memory=True,
                              num_workers=4
                              )

    return train_loader


import time


def train(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    torch.cuda.set_device(args.local_rank)

    net = AlexNet(num_classes=10)

    trainset = cifar_trainset(args.local_rank)
    train_loader = dataset_split(args=args, dataset=trainset)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset
    )

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    criterion = nn.CrossEntropyLoss()

    print(type(train_loader))
    print("My rank is:", local_rank, "The trainloader size is ", len(train_loader))

    grad_portion = args.batch_size / np.sum(args.batch_size)
    if local_rank==0:
        print("The grad_portion is ",grad_portion)

    if local_rank == 0:
        start_time = time.time()

    model_engine.train()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            if target_dtype:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss*grad_portion[local_rank])
            model_engine.step()

            # print statistics
            #print('[Rank %d][EPOCH %d][INDEX %d] The minibatch loss is %.5f' % (local_rank, epoch, i + 1, loss.item()))
            running_loss += loss.item()
            if args.local_rank == 0 and i % args.log_interval == (
                    args.log_interval - 1):  # print every log_interval mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0

    if local_rank == 0:
        end_time = time.time()
        print("The final time cost ia ", end_time - start_time)
    print('Finished Training')
    dist.barrier()


if __name__ == '__main__':
    args = get_args()
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    train(args)
