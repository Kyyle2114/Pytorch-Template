import os
import argparse
import wandb
import torchinfo
import time
import datetime
import json
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn 
import torchvision.transforms as T
import torch.distributed as dist

from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils import misc, datasets
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.lr_sched import CosineAnnealingWarmUpRestarts
from engines import engine_train
from models import cnn

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # initial config 
    parser.add_argument('--port', type=int, default=1234,
                        help='port number for distributed learning')
    
    parser.add_argument('--seed', type=int, default=21, 
                        help='random seed')
    
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    
    # training config 
    parser.add_argument('--dataset_path', type=str, default='./dataset', 
                        help='dataset path')
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='batch size allocated to each GPU')
    
    parser.add_argument('--epoch', type=int, default=10, 
                        help='total training epoch')
    
    parser.add_argument('--patience', type=int, default=50, 
                        help='patience for early stopping')
    
    # optimizer config 
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='initial (base) learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='weight decay')
    
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='epochs to warmup learning rate')
    
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='clip gradient norm (default: None, no clipping)')
    
    # wandb config
    parser.add_argument('--project_name', type=str, default='Model-Training', 
                        help='WandB project name')
    
    parser.add_argument('--run_name', type=str, default='Model-Training', 
                        help='WandB run name')
    
    return parser

def main(rank, args):
    
    """
    Main function for model training.

    Args:
        args (parser): Parsed arguments.
    """
    ### distributed training & WandB setting ###
    misc.seed_everything(args.seed)

    misc.init_distributed_training(rank, args)
    local_gpu_id = args.gpu
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    if misc.is_main_process():
        wandb.init(project=args.project_name)
        wandb.run.name = args.run_name 
        
    ### dataset & dataloader ### 
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # CIFAR10 for testing
    train_set = datasets.make_cifar10_dataset(
        dataset_path=args.dataset_path,
        train=True,
        transform=transform
    )

    val_set = datasets.make_cifar10_dataset(
        dataset_path=args.dataset_path,
        train=False,
        transform=transform
    )
    
    if args.dist:
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True, seed=args.seed)
        batch_sampler_train = BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_loader = DataLoader(
            train_set, 
            batch_sampler=batch_sampler_train, 
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=misc.seed_worker
        )
    
    if not args.dist:
        train_loader = DataLoader(
            train_set, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=misc.seed_worker
        )
    
    val_loader = DataLoader(
            val_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    ### model config ###
    device = torch.device(f'cuda:{local_gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    model = cnn.SimpleCNNforCIFAR10()
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print model info 
    if misc.is_main_process():
        print()
        print('=== MODEL INFO ===')
        torchinfo.summary(model)
        print()

    if args.dist:
        # if multi-gpu training, use SyncBatchNorm
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])    

    ### training config (loss, optimizer, scheduler) ###
    criterion = nn.CrossEntropyLoss().to(device)
    loss_scaler = NativeScaler()

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.abs_lr = args.lr * eff_batch_size / 256
    
    # following timm: set wd as 0 for bias and norm layers
    model_without_ddp = model.module if args.dist else model
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=1e-7 # small lr for warm-up
    )
    print(optimizer)
    print()
    
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, 
        T_0=args.epoch//2, 
        T_mult=1, 
        eta_max=args.abs_lr, 
        T_up=args.warmup_epochs, 
        gamma=0.9
    )

    ### wandb logging ###
    if misc.is_main_process():
        wandb.watch(
            models=model_without_ddp,
            criterion=criterion,
            log='all',
            log_freq=10
        )
    
        wandb.run.summary['optimizer'] = type(optimizer).__name__
        wandb.run.summary['scheduler'] = type(scheduler).__name__
        wandb.run.summary['initial lr'] = args.abs_lr
        wandb.run.summary['total epoch'] = args.epoch
        wandb.run.summary['weight decay'] = args.weight_decay
        wandb.run.summary['batch size'] = args.batch_size
        wandb.run.summary['effective batch size'] = args.batch_size * misc.get_world_size()
    
    ### model training ### 
    start_time = time.time()
    max_accuracy = 0.0  # for evaluation
    max_loss = np.inf   # for evaluation 
    
    # Early stopping : determined based on the validation loss. lower is better (mode='min')
    es = misc.EarlyStopping(patience=args.patience, delta=0, mode='min', verbose=True)
    early_stop_tensor = torch.tensor(0, device=device) if args.dist else None
    
    for epoch in range(args.epoch):
        
        # check early stopping
        if args.dist:
            # Synchronize early stopping decision across all processes
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item() == 1:
                break
        
        if args.dist:
            train_sampler.set_epoch(epoch)
        
        # model train
        train_stats = engine_train.train_one_epoch(
            model=model, 
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args
        )
        
        # save the model 
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epoch):
            try:
                misc.save_model(
                    args=args, 
                    model=model, 
                    model_without_ddp=model_without_ddp, 
                    optimizer=optimizer,
                    loss_scaler=loss_scaler, 
                    epoch=epoch
                )
            except Exception as e:
                print(f"Error saving model: {e}")
            
        scheduler.step()
            
        # model evaluation (validation set)
        if misc.is_main_process():
            eval_stats = engine_train.evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device
            )
            
            print(f"[INFO] Accuracy of the network on the {len(val_set)} test images: {eval_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, eval_stats["acc1"])
            val_loss = eval_stats['loss']
            print(f'[INFO] Current max validation accuracy: {max_accuracy:.2f}%')
            
            if val_loss < max_loss:
                print(f'[INFO] Validation loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save the model.')
                max_loss = val_loss
                try:
                    misc.save_model(
                        args=args, 
                        model=model, 
                        model_without_ddp=model_without_ddp, 
                        optimizer=optimizer,
                        loss_scaler=loss_scaler, 
                        epoch=epoch,
                        is_best=True
                    )
                except Exception as e:
                    print(f"Error saving model: {e}")
            
            # check early stopping
            es(val_loss)
            if es.early_stop:
                print(f'[INFO] Early stopping triggered at epoch {epoch+1} \n')
                if args.dist:
                    # Broadcast early stopping signal to all processes
                    early_stop_tensor = torch.tensor(1, device=device)
                break
            else:
                if args.dist:
                    early_stop_tensor = torch.tensor(0, device=device)

        # stats logging
        if args.output_dir and misc.is_main_process():
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in eval_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # wandb logging
        if misc.is_main_process():
            wandb.log(
                {
                    'Training loss': train_stats['loss'],
                    'Training learning rate': train_stats['lr'],
                    'Evaluation loss': eval_stats['loss'],
                    'Evaluation top-1 accuracy': eval_stats['acc1']
                }, step=epoch+1
            )
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if misc.is_dist_avail_and_initialized:
        dist.destroy_process_group()
    
    
if __name__ == '__main__': 

    wandb.login()
    
    parser = argparse.ArgumentParser('Model-Training', parents=[get_args_parser()])
    args = parser.parse_args() 
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    args.ngpus_per_node = torch.cuda.device_count()
    args.dist = True if args.ngpus_per_node > 1 else False
    args.gpu_ids = list(range(args.ngpus_per_node))
    args.num_workers = args.ngpus_per_node * 4
        
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=args.ngpus_per_node,
        join=True
    )
        
    print('=== Training Complete ===')