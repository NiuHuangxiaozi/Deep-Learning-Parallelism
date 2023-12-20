#!/usr/bin/env python3
import os
import argparse
import torch
import time
import logging
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.distributed import init_process_group, destroy_process_group
from argparse import Namespace
from sampler import Distributed_Elastic_Sampler
from torch.utils.data import DataLoader, Dataset
from model import Resnet_large

def cifar_set(local_rank, dl_path='/home/ainet/wsj/'):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(449),
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
    testset = torchvision.datasets.CIFAR10(root=dl_path,
                                           train=False,
                                           download=True,
                                           transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset, testset


def dataset_split(
        args: Namespace,
        dataset
) -> torch.utils.data.DataLoader:
    sampler_dict = \
        {
            'method': args.data_partition_method,
            'manual_partition_list': args.manual_partition_lists
        }

    sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size[args.global_rank],
                              sampler=sampler,
                              pin_memory=args.train_loader_pin_memory,
                              num_workers=args.train_loader_num_workers
                              )

    return train_loader


def train(args, train_data):
    # prepare for the dataset
    train_loader = dataset_split(args, train_data)
    print('My rank is %d. The length of the train_loader is %d.' % (args.global_rank, len(train_loader)))

    resnet152 = models.resnet152(pretrained=False)
    model = Resnet_large(resnet152).cuda()

    if args.global_rank==0:
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=0.001, )
    criterion = nn.CrossEntropyLoss()

    grad_portion = args.batch_size / np.sum(args.batch_size)

    model.train()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        if args.local_rank == 0: #开始记录时间
            start_time = time.time()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()
            if i == 0:
                print("The input shape is ", inputs.shape)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            (loss * grad_portion[args.global_rank]).backward()

            if (i + 1) % args.gradient_step == 0: #梯度累计够了开始更新参数
                average_gradients(model, args)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            # print('[Rank %d][EPOCH %d][INDEX %d] The minibatch loss is %.5f' % (args.local_rank, epoch, i + 1, loss.item()))

            running_loss += loss.item()
            if i % args.log_interval == (args.log_interval - 1):  # print every log_interval mini-batches
                print('[RANK %d][EPOCH %d][INDEX %d] :Average loss: %.4f' % (
                    args.global_rank, epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0

        if args.local_rank == 0:
            end_time = time.time()
            print('Rank %d The epoch %d time cost is %.5f' % (args.global_rank, epoch, end_time - start_time))

    print('Rank %d finished training' % (args.global_rank))

    return model


def average_gradients(model, args):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


# ////////////////////////////////////////////////////////
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_destroy():
    destroy_process_group()


def Get_args():
    parser = argparse.ArgumentParser(description='Alexnet train on cifar10.')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank always refer to specific gpu.')
    parser.add_argument('--global_rank', default=-1, type=int, help='Global Rank.')
    parser.add_argument('--log_interval', type=int, default=50, help="print average loss per interval")
    parser.add_argument('--gradient_step', type=int, default=10, help="gradient accumulation")
    parser.add_argument('--data_path', type=str, default='./data/', help="Dataset path")
    parser.add_argument('--data_partition_method',
                        type=str,
                        default='manual',
                        help='A method about how to split data between different processes.')
    parser.add_argument('--manual_partition_lists',
                        type=list,
                        default=[40000, 10000],
                        help='different processes\' data proportion')
    parser.add_argument('--batch_size',
                        type=list,
                        default=[80, 20],
                        help='different processes\' data proportion')

    parser.add_argument('--train_loader_shuffle',
                        type=bool,
                        default=True,
                        help='whether use shuffle method in train_dataloader')
    parser.add_argument('--train_loader_pin_memory',
                        type=bool,
                        default=True,
                        help='whether use pin-memory method in train_dataloader')
    parser.add_argument('--train_loader_num_workers',
                        type=int,
                        default=2 * 3,  # 2 *(device count)
                        help='how many process to help load data(use in train_dataloader.')

    args = parser.parse_args()
    return args


def main():
    ddp_setup()
    args = Get_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.global_rank = int(os.environ["RANK"])
    if args.local_rank == 0:
        print("The config is :", args)
    train_data, test_data = cifar_set(args.local_rank,args.data_path)
    model = train(args, train_data)
    print("RANK :", args.global_rank, "All Finished")
    ddp_destroy()


# /////////////////////////////////////////////////////////////////


if __name__ == '__main__':
    main()
