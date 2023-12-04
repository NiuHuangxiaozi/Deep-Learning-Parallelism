#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=1 train.py --log_interval=200 --epochs=1 --backend='nccl'  --train_loader_num_workers=2