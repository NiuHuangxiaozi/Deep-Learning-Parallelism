import os
import sys
import argparse
from argparse import Namespace
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed import init_process_group, destroy_process_group

# ---- GPU check ------------
min_gpu_count = 2

if torch.cuda.device_count() < min_gpu_count:
    print(f"Unable to locate sufficient {torch.cuda.device_count()} gpus to run this example. Exiting.")
    sys.exit()


# ---------------------------
class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 320000)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(320000, 500)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


def tp_setup():
    init_process_group(backend="nccl")
    # 这句话的作用是设置当前进程使用的 CUDA 设备。
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def tp_destroy():
    destroy_process_group()


def main(
        args: Namespace
) -> None:
    tp_setup()
    test(args)
    tp_destroy()


def test(
        args: Namespace
) -> None:
    device_mesh = DeviceMesh("cuda", torch.arange(args.world_size))
    rank = device_mesh.get_rank()
    print(f"Starting PyTorch TP example on rank {rank}.")
    if rank == 0:
        assert (args.world_size % 2 == 0), f"TP examples require even number of GPUs, but got {args.world_size} gpus"
        print("GPU Mesh:", device_mesh)
    # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.

    tp_model = ToyModel().to("cuda")
    optimizer = torch.optim.AdamW(tp_model.parameters(), lr=args.lr, foreach=True)
    tp_model = (parallelize_module
        (
        module=tp_model,
        device_mesh=device_mesh,
        parallelize_plan={
            "in_proj": ColwiseParallel(),
            "out_proj": RowwiseParallel(),
        },
    ))

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    if rank == 0:
        print(rank, ":Tensor Parallel training starting...")

    for i in range(args.epoch):
        # For TP, input needs to be same across all TP ranks.
        # Setting the random seed is to mimic the behavior of dataloader.
        torch.manual_seed(i)
        inp = torch.rand(20, 10, device="cuda")
        output = tp_model(inp)
        output.sum().backward()
        optimizer.step()
        print(rank, f"Tensor Parallel iter {i} completed")

    if rank == 0:
        print(rank, ":Tensor Parallel training completed!")


import pprint

# 使用torchrun进行启动
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple test on tensor parallelism.')
    parser.add_argument('--world_size', type=int, help="number of GPU.")
    parser.add_argument('--epoch', type=int, help="times to train.")
    parser.add_argument("--lr", type=float, help="learning rate.")
    args = parser.parse_args()
    print("This is the parameter you choose:")
    pprint.pprint(args)
    main(args)