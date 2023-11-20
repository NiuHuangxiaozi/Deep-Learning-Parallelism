## 一、实验内容

这个实验讲了如何在四块gpu上使用数据并行和流水线并行进行Transformer模型的训练，实验内容来自pytorch官方的tutorial

[Training Transformer models using Distributed Data Parallel and Pipeline Parallelism — PyTorch Tutorials 2.1.1+cu121 documentation](https://pytorch.org/tutorials/advanced/ddp_pipeline.html)

## 二、实验配置

我是在锯齿云上面直接找了四张显卡，然后选择了pytorch1.12的镜像，具体配置如下：

```python
pytorch 1.12
Ubuntu 18.04
Python 3.9
CUDA 11.3
cuDNN 8
NVCC 
Pytorch 1.12.0
torchvision 0.13.0
torchaudio 0.12.0
torchtext 0.13.0
torchdata 0.4.0
```

