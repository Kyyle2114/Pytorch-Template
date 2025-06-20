#!/bin/bash

# set CUDA device
export CUDA_VISIBLE_DEVICES=0

# run training
python main.py \
    --output_dir ./output_dir \
    --dataset_path ./dataset \
    --batch_size 32 \
    --epoch 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --patience 20 \
    --project_name "CIFAR10-Training" \
    --run_name "SimpleCNN-CIFAR10" 