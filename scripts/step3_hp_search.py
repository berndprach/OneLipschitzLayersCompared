import math
import random

import torch
import yaml

from src import data, metrics, run_logging
from src.optimizer import OneCycleSGD, OneCycleSGDHp
from src.run_logging import Timer
from src.trainer import Trainer

from . import constants
from .step1_measure_batch_time import get_model_from_idx
from .util import convert_arguments_from_strings

SEP = ";"


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    model = get_model_from_idx(idx)
    data_preprocessor = get_data_preprocessor(dataset_name)

    lr = 10 ** random.uniform(-4, -1)
    wd = 10 ** random.uniform(-5.5, -3.5)
    hp = {"lr": lr, "wd": wd, "nrof_hours": nrof_hours}

    epochs = get_epoch_budget(idx, dataset_name, nrof_hours)
    print(f"Training model {idx} for {epochs} epochs.")

    loss_function = metrics.OffsetXent(
        offset=constants.LOSS_OFFSET,
        temperature=constants.LOSS_TEMPERATURE
    )
    optimizer = get_optimizer(model, lr, wd, epochs)
    trainer = Trainer(
        model, loss_function, optimizer, get_metrics(loss_function)
    )
    train(trainer, data_preprocessor, epochs)

    final_val_stats = trainer.evaluate(data_preprocessor.val)
    with open(constants.HP_SEARCH_RESULTS_FILE, "a") as f:
        f.write(f"{idx}{SEP} {hp}{SEP} {final_val_stats}\n")


def get_data_preprocessor(dataset_name):
    dataset = getattr(data.datasets, dataset_name)()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return data.get_dp(
        dataset, constants.BATCH_SIZE, device, val_proportion=0.1
    )


def get_epoch_budget(idx, dataset_name, nrof_hours):
    fp = constants.EPOCH_BUDGETS_2H_FILES[dataset_name]
    with open(fp, "r") as f:
        epoch_budgets_2h = yaml.load(f, Loader=yaml.SafeLoader)
    epoch_budget = epoch_budgets_2h[idx] * nrof_hours // 2
    return int(math.floor(epoch_budget))


def get_optimizer(model, lr, wd, epochs):
    hp = OneCycleSGDHp(peak_lr=lr, weight_decay=wd)
    return OneCycleSGD(
        model.parameters(),
        total_steps=epochs,
        hp=hp,
    )


def get_metrics(loss):
    return [
        loss,
        metrics.Accuracy(),
        metrics.BatchVariance(),
        metrics.Margin(),
        metrics.CRA(1 * 36 / 255),
        metrics.CRA(2 * 36 / 255),
        metrics.CRA(3 * 36 / 255),
        metrics.CRA(1),
    ]


def train(trainer, data_preprocessor, epochs):
    timer = Timer()
    line_formatter = run_logging.DoubleColumnsLineFormatter(6)
    timer.start()

    for epoch in range(1, epochs+1):
        train_stats = trainer.train_epoch(data_preprocessor.train)
        val_stats = trainer.evaluate(data_preprocessor.val)

        all_stats = {
            "Epoch": epoch,
            "AvgSec": timer.seconds_elapsed / epoch,
            **train_stats,
            **val_stats
        }
        print(line_formatter(all_stats))
        trainer.optimizer.scheduler_step()


