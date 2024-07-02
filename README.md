# Deep Learning Model Parallelism

------



# 1. Introduction

------



This repository documents my journey in learning about deep learning parallelism, including Distributed Data Parallel (DDP), tensor parallelism, pipeline parallelism, and more. I have gathered a variety of materials from sources such as CSDN, PyTorch tutorials, and GitHub repositories. I am thrilled to share this resource, and I hope it will be of help to you, even if just a little.



# 2. Details

------



Here is a brief summary of the repository:

| experiment                                   | short description                                     | related info       |
| -------------------------------------------- | ----------------------------------------------------- | ------------------ |
| Alexnet_deepspeed_parallelism (One Node)     | different partitions  on DeepSpeed                    | pytorch, deepspeed |
| Alexnet_pytorch_UnbanlancedDDP_parallelism   | different partitions on my gradient  communication    | pytorch            |
| ResNet_parameter_server_parallelism          | parameter sever utilizing torch.rpc                   | pytorch rpc        |
| Resnet50_OneNode_Pipline_parallelism         | pipeline parallelism using torch.rpc                  | pytorch, rpc       |
| ResNet152_multinode_DDP_parallelism          | multinode DDP                                         | pytorch            |
| ring_allreduce                               | ring topology allreduce                               | pytorch            |
| Rnn_parameter_server_parallelism             | ps using torch rpc                                    | pytorch, rpc       |
| Tonymodel_multinode_DDP_torchrun_parallelism | pytorchDDP towards torchrun                           | pytorch            |
| TonyModel_tensor_parallelism                 | tensor parallelism utilizing DTensor                  | pytorch, DTensor   |
| Transformer_data_pipeline_parallelism        | data parallelism integrated with pipeline parallelism | pytorch, rpc       |
| Transformer_pipeline_parallelism             | pipeline parallelism for Transformer on two node      | pytorch,  rpc      |















