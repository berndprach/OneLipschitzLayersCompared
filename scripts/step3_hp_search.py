import math
import random

import yaml

from src import metrics
from src.data.get_data_preprocessor import get_data_preprocessor
from src.optimizer import OneCycleSGD
from src.trainer import Trainer, train_model
from src.models.model_layer_combinations import get_model_by_idx

from . import constants as c
from .util import convert_arguments_from_strings


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    model = get_model_by_idx(idx)
    data_preprocessor = get_data_preprocessor(dataset_name, c.BATCH_SIZE, 0.1)
    epochs = get_epoch_budget(idx, dataset_name, nrof_hours)
    print(f"Training model {idx} for {epochs} epochs.")

    lr = 10 ** random.uniform(-4, -1)
    wd = 10 ** random.uniform(-5.5, -3.5)
    hp = {"lr": lr, "wd": wd, "nrof_hours": nrof_hours}

    final_val_stats = evaluate_hps(model, data_preprocessor, epochs, lr, wd)

    fp = c.get_hp_search_results_path(dataset_name)
    with open(fp, "a") as f:
        f.write(f"{idx}{c.SEPERATOR}{hp}{c.SEPERATOR}{final_val_stats}\n")
    print(f"Append hp search results to {fp}.")


def evaluate_hps(model, data_preprocessor, epochs, lr, wd):
    trainer = get_trainer(epochs, model, lr, wd)
    train_model(trainer, data_preprocessor, epochs)
    final_val_stats = trainer.evaluate(data_preprocessor.val)
    return final_val_stats


def get_trainer(epochs, model, lr, wd):
    loss_function = metrics.OffsetXent(
        offset=c.LOSS_OFFSET,
        temperature=c.LOSS_TEMPERATURE
    )
    optimizer = OneCycleSGD(
        model.parameters(),
        total_steps=epochs,
        peak_lr=lr,
        weight_decay=wd
    )
    used_metrics = [loss_function] + c.METRICS
    return Trainer(model, loss_function, optimizer, used_metrics)


def get_epoch_budget(idx, dataset_name, nrof_hours):
    epoch_budgets_2h = get_epoch_budgets_2h(dataset_name)
    epoch_budget = epoch_budgets_2h[idx] * nrof_hours // 2
    return int(math.floor(epoch_budget))


def get_epoch_budgets_2h(dataset_name):
    fp = c.EPOCH_BUDGETS_2H_FILES[dataset_name]
    print(f"Reading epoch budgets from {fp}.")
    with open(fp, "r") as f:
        epoch_budgets_2h = yaml.load(f, Loader=yaml.SafeLoader)
    return epoch_budgets_2h

