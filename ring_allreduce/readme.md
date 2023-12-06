# Ring Allreduce的简单实现

### 一、实验内容

本次实验想法来源于pytorch tutorial官方教程《WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH》，网址是：[Writing Distributed Applications with PyTorch — PyTorch Tutorials 2.1.1+cu121 documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

使用点对点的通信原语实现与节点数量无关的allreduce操作。

这个实验的原理来源于：[Ring Allreduce - 简书 (jianshu.com)](https://www.jianshu.com/p/8c0e7edbefb9)

### 二、实验环境

这个是在锯齿云上租了4张A16跑的。

```python
Ubuntu20.04
Python 3.10
Pytorch 2.0.1
CUDA 11.8
cuDNN 8
NVCC
```

### 三、实验感受

目前经过实验，在4张gpu上成功的实现了allreduce操作，实验的例子就写在main函数里面。目前遗留了一个后续开发的问题（因为多进程的编程还是困难的）：

1、能不能使用isend和irecv等非阻塞的原语函数。在代码中我们使用的是send，recv等函数，这些是阻塞型的函数，就是等待对方响应完后才能继续执行。自己试过一个和pytorch官方一样的版本，但是死锁了。