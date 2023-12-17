# 使用Resnet152模型和cifar10增强数据集在多节点上进行分布式训练



## 一、项目介绍

这是一个学习pytorch的数据并行实验使用resnet152模型，并自己实现了数据并行中的all_reduce操作和梯度累计操作。

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

