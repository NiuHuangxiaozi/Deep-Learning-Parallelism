#!/bin/bash
deepspeed  train.py --deepspeed_config=ds_config.json --epochs=200