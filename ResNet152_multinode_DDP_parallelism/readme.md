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

## 三、实验收获

#### 3.1学会了梯度累计

```python
if (i + 1) % args.gradient_step == 0: #梯度累计够了开始更新参数
                average_gradients(model, args)
                optimizer.step()
                optimizer.zero_grad()
```

通过上面的代码我们能够自己控制多少轮进行一次梯度交换。

#### 3.1未来的展望。

因为现在使用的是dist.all_reduce所以以后看一看能不能融合ring-allreduce和实现异步all_reduce。
