

import math
from typing import TypeVar, Optional, Iterator
import torch
import torch.distributed as dist
from torch.utils.data import Dataset,Sampler


#  when you use command "from sampler import *" The DistributedSampler will be imported.
__all__ = ["Distributed_Elastic_Sampler", ]


T_co = TypeVar('T_co', covariant=True)



class Distributed_Elastic_Sampler(Sampler[T_co]):
    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 partition_strategy:dict=None
                 ) -> None:
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
        self.partition_strategy=partition_strategy
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self.num_samples=None
        self.total_size=None
        if self.partition_strategy["method"]=='even':
            self.even_partition()
        else:
            self.auto_partition()
        self.shuffle = shuffle
        self.seed = seed
    def auto_partition(self):
        if self.partition_strategy["method"]=="manual":
            self.num_samples= self.partition_strategy["manual_partition_list"]
            assert self.num_replicas == len(self.num_samples), str(self.num_replicas)+" not equal to "+str(len(self.num_samples))
            self.total_size=sum(self.num_samples)
            assert self.total_size==len(self.dataset), str(self.total_size)+" not equal to "+str(len(self.dataset))
        else:
            assert(self.partition_strategy["method"]=="manual")

    def even_partition(self):
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

        self.num_samples =[self.num_samples]*self.num_replicas

    def even_iter(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
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
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def auto_iter(self):
        if self.partition_strategy['method']=='manual':
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))

            assert len(indices) == self.total_size

            # subsample
            pre_index = 0
            indices_list = []
            for val in self.num_samples:
                indices_list.append(indices[pre_index:pre_index + val])
                pre_index += val
            # subsample
            for indices in indices_list:
                assert len(indices_list[self.rank]) == self.num_samples[self.rank]
            return iter(indices_list[self.rank])


    def __iter__(self) -> Iterator[T_co]:
        if self.partition_strategy['method']=='even':
            return self.even_iter()
        else:
            return self.auto_iter()
    def __len__(self) -> int:
        return self.num_samples[self.rank]

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
