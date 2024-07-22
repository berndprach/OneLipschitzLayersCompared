import functools
import itertools
import time

import numpy as np
import torch

LR = 1e-3


def add_unit(time_in_seconds: float):
    if time_in_seconds < 0:
        return "-" + add_unit(-time_in_seconds)
    if time_in_seconds < 1:
        return f"{time_in_seconds * 1000:.2f} ms"
    if time_in_seconds > 60**2:
        return f"{time_in_seconds / 60**2:.2f} h"
    if time_in_seconds > 60:
        return f"{time_in_seconds / 60:.2f} min"
    return f"{time_in_seconds:.2f} sec"


def get_training_mean(model, loader, nrof_batches):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    # Warm up:
    first_batch = next(iter(loader))
    out = model(first_batch[0])
    out.sum().backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    start_time = time.time()

    for i, (batch_x, _) in enumerate(itertools.cycle(loader)):
        out = model(batch_x)
        out.sum().backward()
        optimizer.step()
        
        if i+1 == nrof_batches:
            break

    
    torch.cuda.synchronize()
    training_sum = time.time() - start_time
    training_mean_seconds = training_sum / nrof_batches
    return training_mean_seconds


def get_validation_mean(model, val_loader, nrof_batches, cached=True):
    model.eval()

    model_forward = functools.partial(forward_pass, model)

    if cached:
        model_forward = torch.nn.utils.parametrize.cached()(model_forward)
    
    # Warm up and cache:
    first_batch = next(iter(val_loader))
    model_forward(first_batch[0])

    torch.cuda.synchronize()
    start_time = time.time()

    for i, (batch_x, _) in enumerate(itertools.cycle(val_loader)):
        model_forward(batch_x)
        
        if i+1 == nrof_batches:
            break
    
    torch.cuda.synchronize()
    val_sum = time.time() - start_time
    val_mean_seconds = val_sum / nrof_batches
    return val_mean_seconds



@torch.no_grad()
def forward_pass(model, batch):
    out = model(batch)
    out.sum()


def backward_pass(model, optimizer, batch, device="cuda"):
    batch = batch.to(device)
    out = model(batch)
    out.sum().backward()
    optimizer.step()





def evaluate_all_model_time_statistics(model: torch.nn.Module,
                                       train_loader,
                                       test_loader,
                                       nrof_batches: int = 100,
                                       ):

    return {
        "train_mean": get_training_mean(model, train_loader, nrof_batches),
        "test_cached_mean": get_validation_mean(model, test_loader, nrof_batches, cached=True),
    }
