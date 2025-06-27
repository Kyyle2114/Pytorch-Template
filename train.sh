#!/bin/bash

# Set which GPUs to use (e.g., "0,1" or "0,1,2,3").
# The script will automatically count them to set the number of processes for torchrun.
# If this variable is not set, torch will use all available GPUs.
export CUDA_VISIBLE_DEVICES=0,1

# Automatically determine the number of GPUs from CUDA_VISIBLE_DEVICES
# or all available GPUs if the variable is not set.
N_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')

echo "Using $N_GPUS GPUs based on CUDA_VISIBLE_DEVICES..."

# run training using torchrun
torchrun --nproc_per_node=$N_GPUS main.py \
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