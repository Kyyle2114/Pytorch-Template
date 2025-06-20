# Pytorch Template

Structured and modular template for deep learning model training.

## Deep Learning Model Training Template

This repository provides a structured and modular template for deep learning model training, inspired by the training pipelines of [MAE](https://github.com/facebookresearch/mae) from Facebook Research (Meta).

### Features
- Modularized training and evaluation pipeline
- Support for distributed training (DistributedDataParallel)
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

### Running the Training
1. Execute the training script using `train.sh`:
```bash
sh train.sh
```

2. All available training arguments can be found at the top of `main.py` in the `get_args_parser()` function. These include:
   - Training configuration (batch size, epochs, etc.)
   - Optimizer settings (learning rate, weight decay, etc.)
   - Dataset paths
   - WandB logging settings
   - And more...

### Current Implementation
The current implementation includes:
- A simple CNN model for CIFAR10 classification
- Distributed training support
- Early stopping
- Learning rate scheduling with linear warmup
- Model checkpointing

The code is designed to be modular, allowing you to easily:
- Replace the model architecture in `models/`
- Change the dataset in `dataset/` and `utils/datasets.py`
- Modify training configurations
- Add new features while maintaining the existing pipeline structure

### Cursor Rules 

The cursor rules for this repository are inspired by the collection of best practices at [awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules).