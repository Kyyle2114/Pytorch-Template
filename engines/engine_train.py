import math
import sys
from typing import Iterable
from timm.utils import accuracy

import torch

from utils import misc

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int, 
    loss_scaler,
    args=None) -> dict:
    """
    Train the model for one epoch. 

    Args:
        model (torch.nn.Module): PyTorch Model.
        data_loader (Iterable): PyTorch DataLoader.
        criterion (torch.nn.Module): PyTorch loss function.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        device (torch.device): Device to use. 
        epoch (int): Current epoch.
        loss_scaler (GradScaler): Loss scaler for mixed precision training.
        args (parser, optional): Parsed argument. Defaults to None.

    Returns:
        dictionary: contains the global average for each metric, such as training loss and learning rate 
    """
    model.train(True)
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    accum_iter = args.accum_iter
    
    criterion = criterion
    optimizer.zero_grad()
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss, 
            optimizer,
            clip_grad=args.clip_grad, 
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)
    print()
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: torch.nn.Module,
    device: torch.device
    ) -> dict:
    """
    Evaluate the model. 

    Args:
        model (torch.nn.Module): PyTorch Model.
        data_loader (Iterable): PyTorch DataLoader.
        criterion (torch.nn.Module): PyTorch loss function.
        device (torch.device): Device to use. 

    Returns:
        dict: contains the global average for each metric, such as validation accuracy
    """
    criterion = criterion

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Evaluate:'

    # switch to evaluation mode
    model.eval()

    for (images, targets) in metric_logger.log_every(data_loader, 50, header):
        
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
            
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}