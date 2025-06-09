# Code Template for DL training 

Structured and modular template for deep learning model training.

## Deep Learning Training Template

This repository provides a structured and modular template for deep learning model training, inspired by the training pipelines of [MAE](https://github.com/facebookresearch/mae) from Facebook Research (Meta).

### Features
- Modularized training and evaluation pipeline
- Support for distributed training (DDP)
- Configurable hyperparameters via argument parsing
- Easy integration with different datasets and models
- Logging and visualization with Weights & Biases (WandB)

## Getting Started

### Installation
```bash
conda create -n DL python=3.10
conda activate DL
pip install -r requirements.txt
```