## 一、实验内容

这个实验是使用torch.tpc实现参数服务器，使用的模型是resnet系列，我自己测试在单机多卡运行正常，但是多机多卡代码init_rpc会报错"Connection Refused"  实验代码是来自[xbfu/PyTorch-ParameterServer: An implementation of parameter server framework in PyTorch RPC. (github.com)](https://github.com/xbfu/PyTorch-ParameterServer/tree/main)

## 二、实验环境

```python
Ubuntu20.04
Python 3.10
Pytorch 2.0.1
CUDA 11.8
cuDNN 8
NVCC
torchvision 0.15.2+cu118
```

