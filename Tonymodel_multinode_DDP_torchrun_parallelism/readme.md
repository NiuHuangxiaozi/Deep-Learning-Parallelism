## 一、实验内容

使用torchrun的命令实现多个节点的数据并行模型的并行训练，模型为随便的小模型，主要资料来源于：

1、[examples/distributed/ddp-tutorial-series at main · pytorch/examples (github.com)](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)

2、[Multinode Training — PyTorch Tutorials 2.1.1+cu121 documentation](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)

## 二、实验环境

```python
Pytorch 2.0.1
Ubuntu20.04
Python 3.10
CUDA 11.8
cuDNN 8
NVCC
```

两台机器都是上面的配置，都各有一块GPU，输入的命令是：

rank 0主节点：

```python
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 multinode_torchrun.py 50 10 
```

其他节点：

```python
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 multinode_torchrun.py 50 10 
```

