#/usr/bin/bash

CUDA_VISIBLE_DEVICES=1 accelerate launch --multi_gpu --num_processes 1 \
    --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small.py \
    --nnet_path=cifar10_uvit_small.pth
