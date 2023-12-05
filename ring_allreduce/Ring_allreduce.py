import torch
import os
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

""" Implementation of a ring-reduce with addition. """


def ring_allreduce(para_tensor):
    rank = dist.get_rank()
    size = dist.get_world_size()
    blocks = list(para_tensor.chunk(size))
    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    send_block_index = rank
    recv_block_index = ((rank - 1) + size) % size

    # Scatter-Reduce
    for i in range(size - 1):
        buff = blocks[send_block_index].clone()
        buff1 = blocks[send_block_index].clone()
        if rank % 2 == 0:
            dist.send(buff, right)
            dist.recv(buff1, left)
            blocks[recv_block_index] += buff1
        else:
            dist.recv(buff, left)
            dist.send(buff1, right)
            blocks[recv_block_index] += buff
        send_block_index = ((send_block_index - 1) + size) % size
        recv_block_index = ((recv_block_index - 1) + size) % size


def setup():
    init_process_group(backend="nccl")
    # 这句话的作用是设置当前进程使用的 CUDA 设备。
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def destroy():
    destroy_process_group()


def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        tmp = torch.Tensor([1, 1, 1, 1]).cuda()
    elif local_rank == 1:
        tmp = torch.Tensor([2, 2, 2, 2]).cuda()
    elif local_rank == 2:
        tmp = torch.Tensor([3, 3, 3, 3]).cuda()
    elif local_rank == 3:
        tmp = torch.Tensor([4, 4, 4, 4]).cuda()

    print(tmp.shape)
    allreduce(tmp)
    print("My rank is ", local_rank, "The tmp is", tmp)
    destroy()


if __name__ == "__main__":
    main()
