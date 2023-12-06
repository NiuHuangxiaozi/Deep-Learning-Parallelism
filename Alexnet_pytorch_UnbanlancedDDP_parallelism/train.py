"""run.py:"""
# !/usr/bin/env python
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import logging
from argparse import Namespace
from torch.distributed import init_process_group, destroy_process_group
import torchvision
import torchvision.transforms as transforms
from sampler import Distributed_Elastic_Sampler
from torch.utils.data import DataLoader, Dataset
from alexnet import AlexNet
import torch.optim as optim
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="DDP parallelism training based on pytorch")
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help="identify which process")
    parser.add_argument('--log_interval',
                        type=int,
                        default=100,
                        help="print average loss per interval")
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="total training epoch")
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='select a process beckend,sush as gloo,nccl')
    parser.add_argument('--data_partition_method',
                        type=str,
                        default='manual',
                        help='A method about how to split data between different processes.')
    parser.add_argument('--manual_partition_list',
                        type=list,
                        default=[40000, 10000],
                        help='different processes\' data proportion')

    parser.add_argument('--batch_size',
                        type=list,
                        default=[40, 10],
                        help='batchsize list')
    parser.add_argument('--test_batch_size',
                        type=list,
                        default=32,
                        help='test batchsize ')
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
                        default=2 * 2,  # 2 *(device count)
                        help='how many process to help load data(use in train_dataloader.')
    args = parser.parse_args()
    return args


def train(
        args: Namespace,
        train_data
) -> None:
    # prepare for the dataset
    train_loader = dataset_split(args, train_data)
    logging.info(f'My rank is {args.local_rank}. The length of the train_loader is {len(train_loader)}.')

    torch.cuda.set_device(args.local_rank)
    model = AlexNet(num_classes=10).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, )
    criterion = nn.CrossEntropyLoss()

    grad_portion = args.batch_size / np.sum(args.batch_size)

    model.train()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if args.local_rank == 0:
            start_time = time.time()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            (loss * grad_portion[args.local_rank]).backward()
            average_gradients(model, args)
            optimizer.step()

            # print statistics
            # print('[Rank %d][EPOCH %d][INDEX %d] The minibatch loss is %.5f' % (args.local_rank, epoch, i + 1, loss.item()))

            running_loss += loss.item()
            if i % args.log_interval == (args.log_interval - 1):  # print every log_interval mini-batches
                print('[RANK %d][EPOCH %d][INDEX %d] :Average loss: %.4f' % (
                args.local_rank, epoch + 1, i + 1, running_loss / args.log_interval))
                running_loss = 0.0

        if args.local_rank == 0:
            end_time = time.time()
            logging.info(f'The epoch {epoch} time cost is {end_time - start_time}')

    logging.info(f'Rank {args.local_rank} finished training')

    return model


def test(
        args: Namespace,
        test_data,
        model
) -> None:
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=2)
    correct = 0
    total = 0
    local_device = torch.device('cuda:%d' % (args.local_rank))
    model = model.to(local_device)
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.to(local_device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(local_device)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))


def main():
    ddp_setup()
    args = get_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    print("Current config:", args)
    train_data, test_data = cifar_set(args.local_rank)
    model = train(args, train_data)

    if args.local_rank == 0:
        test(args, test_data, model)
    print("RANK :", args.local_rank, "All Finished")
    ddp_destroy()


def ddp_setup():
    init_process_group(backend="nccl")
    # 这句话的作用是设置当前进程使用的 CUDA 设备。
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_destroy():
    destroy_process_group()


def cifar_set(local_rank, dl_path='/tmp/cifar10-data'):
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
            'manual_partition_list': args.manual_partition_list
        }

    sampler = Distributed_Elastic_Sampler(dataset=dataset, partition_strategy=sampler_dict)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=args.batch_size[args.local_rank],
                              sampler=sampler,
                              pin_memory=args.train_loader_pin_memory,
                              num_workers=args.train_loader_num_workers
                              )

    return train_loader


""" Gradient averaging. """


def average_gradients(model, args):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


if __name__ == "__main__":
    main()
