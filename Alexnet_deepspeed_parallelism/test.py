import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sampler import Distributed_Elastic_Sampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
def cifar_trainset(dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    return trainset


def dataset_split(
            dataset: torch.utils.data.Dataset
                   ):

    sampler_dict=\
        {
            'method':'manual',
            'manual_partition_list':[50000]
        }
    sampler=Distributed_Elastic_Sampler(dataset=dataset,partition_strategy=sampler_dict)
    train_loader=DataLoader(dataset=dataset,batch_size=4,shuffle=False,sampler=sampler)
    print(len(train_loader))
def main():
    dist_setup()
    traindata = cifar_trainset()
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

