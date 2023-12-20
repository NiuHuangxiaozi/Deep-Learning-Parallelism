# 使用Resnet152模型和cifar10增强数据集在多节点上进行分布式训练



## 一、项目介绍

这是一个学习pytorch的数据并行实验使用resnet152模型，并自己借鉴pytorch tutorial实现了数据并行中的all_reduce操作和梯度累计操作。

[Writing Distributed Applications with PyTorch — PyTorch Tutorials 2.2.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

## 二、实验环境

实验一：一个服务器含有一张V100，一个服务器含有一张A2000

实验的配置如下：

```python
Ubuntu20.04
Python 3.10
Pytorch 2.0.1
CUDA 11.8
cuDNN 8
NVCC
```

## 三、实验收获

#### 3.1  学会了梯度累计

```python
if (i + 1) % args.gradient_step == 0: #梯度累计够了开始更新参数
                average_gradients(model, args)
                optimizer.step()
                optimizer.zero_grad()
```

通过上面的代码我们能够自己控制多少轮进行一次梯度交换。

#### 3.2  因为我们要测试异构GPU的数据划分能力，所以我们在单机多卡和多级多卡之间做了实验。

详细的实验细节在本文件夹下的pdf能看到。

#### 3.3  未来的展望。

因为现在使用的是dist.all_reduce所以以后看一看能不能融合ring-allreduce和实现异步all_reduce。

#### 3.4  目前存在的问题。

在做实验的过程中，在实验两台机器三张卡的时候，我们发现如果让只有一张卡的机器当作主节点的话，会出现奇怪的bug：在all_reduce的时候，单张卡的机器无法同步，直接越过了dist.all_reduce的同步操作；而拥有两张卡的机器会卡在dist.all_reduce上。但是上面的bug当我指定拥有两张卡的节点为主节点时就消失了。虽然这样不影响实验，但是这里确实留下了一个疑问，静待以后解答。