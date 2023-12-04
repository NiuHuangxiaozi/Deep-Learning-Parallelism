import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sampler import Distributed_Elastic_Sampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
import torch.distributed as dist
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
            dataset: torch.utils.data.Dataset
                   ):
    local_rank=int(os.environ["LOCAL_RANK"])
    sampler_dict=\
        {
            'method':'manual',
            'manual_partition_list':[20000,15000,10000,5000]
        }
    sampler=Distributed_Elastic_Sampler(dataset=dataset,partition_strategy=sampler_dict)
    train_loader=DataLoader(dataset=dataset,batch_size=4,shuffle=False,sampler=sampler)
    print("My rank is:",local_rank,"The trainloader size is ",len(train_loader))
    for index,data in enumerate(train_loader):
        print("My rank is:",local_rank,"The first data shape is",data[0].shape)
        print("My rank is:",local_rank,"The first label shape is",data[1].shape)
        break
def main():
    dist_setup()
    local_rank=int(os.environ['LOCAL_RANK'])
    traindata = cifar_trainset(local_rank)
    dataset_split(traindata)
    dist_detroy()
def dist_setup():
    init_process_group(backend="nccl")
    # 这句话的作用是设置当前进程使用的 CUDA 设备。
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def dist_detroy():
    destroy_process_group()

if __name__=="__main__":
    main()

