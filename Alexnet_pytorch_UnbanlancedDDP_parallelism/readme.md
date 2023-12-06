# 基于pytorch的Batchsize,Datasize任意划分的本地分布式训练实验

### 一、实验内容

本次实验主要是为了实现在DDP的框架上，实现不同的GPU采用不同的batch和dataset数据集的划分，这里面我参考了很多的资料，主要资料如下：

1、deepseed源码。deepspeed/runtime/dataloader.py 这里可以看到deepseed使用的Sampler。[DeepSpeed/deepspeed/runtime/dataloader.py at master · microsoft/DeepSpeed (github.com)](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/dataloader.py)

2、pytorch官方的tutorial，里面介绍了基本的通信原语。[Writing Distributed Applications with PyTorch — PyTorch Tutorials 2.1.1+cu121 documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

3、魔改的模型是来自deepseed的官方example[DeepSpeedExamples/training/cifar at master · microsoft/DeepSpeedExamples (github.com)](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/cifar)

4、pytorch官方DistributedSamper的实现。[torch.utils.data.distributed — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler)

### 二、实验环境

本次实验是在锯齿云上租了4张A16 GPU运行的

```python
Ubuntu20.04
Python 3.10
Pytorch 2.0.1
CUDA 11.8
cuDNN 8
NVCC
```

### 三、实验收获。

#### 1、深刻理解了pytorch的DistributedSampler的工作原理。

下面展示官方DistributedSampler的实现。

```python
import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset
import torch.distributed as dist


#这里说明在别人引用 from xxx.py import * 就会暴露list里面的接口
__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)

class DistributedSampler(Sampler[T_co]):

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator() #pytorch里面产生一个随机数
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        #这一句是最重要的，可以看到是等间隔的划分
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

```

我的改进在代码里有详细的解释。

#### 2、理解了torch.utils.data.Dataloader的一些参数的影响。

本文讨论的影响是：num_workers参数

详细可以看这一篇博客

[Pytorch dataloader中的num_workers (选择最合适的num_workers值)_dataloader num_workers_hxxjxw的博客-CSDN博客](https://blog.csdn.net/hxxjxw/article/details/119531239)

简要来说就是有几个num_workers，就有几个进程在负责将数据加载进RAM。加载数据的进程多了，自然GPU拿到数据的速度也就快了，计算也就快了。一般来说，num_worker设置的太小或者batchsize设置的太小会导致gpu等待数据的情况。

