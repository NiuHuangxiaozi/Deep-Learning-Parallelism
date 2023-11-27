# DISTRIBUTED PIPELINE PARALLELISM USING RPC



## 一、实验目的

这个是pytorch的教材上面的一个例子，使用rpc实现resnet50在单机两张显卡上的流水线并行训练
https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html?highlight=model%20parallel%20best%20practices

值得注意的部分是：

1、rpc.remote函数 返回RRef ，使用to_here卸载到本地

2、rpc.rpc_async()函数 返回 Future ，使用torch.futures.wait_all(out_futures)等待所有的样例处理完毕

3、在使用了rpc分布式训练框架后，要使用dist_autograd和dist_optimizer,dist_autograd里面使用了一种fast mode来解决分布式系统backward的计数依赖问题。

参考的资料为：

[Autograd mechanics — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/notes/autograd.html#autograd-mechanics)

[Distributed Autograd Design — PyTorch 2.1 documentation](https://pytorch.org/docs/stable/rpc/distributed_autograd.html#distributed-autograd-design)

4、因为rpc.remote和rpc.rpc_async()都是异步的操作，所以我们使用了

```python
import threading
self._lock = threading.Lock()
```

来防止多线程同时访问同一个forward模块

## 二、分类

单机流水线并行

## 二、实验环境

```python
Ubuntu20.04
Python 3.9
Pytorch 2.0
CUDA 11.7
cuDNN 8
NVCC
```

