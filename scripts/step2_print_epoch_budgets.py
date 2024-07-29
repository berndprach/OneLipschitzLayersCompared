import math
import os

import yaml

from src import data

from . import constants as c
from .constants import get_batch_times_path, BATCH_SIZE

BUDGET_SECONDS = 2 * 60 * 60  # 2 hours
# NROF_BATCHES_IN_EPOCH = int(math.ceil(50_000 / BATCH_SIZE))


def main(dataset_name: str = "CIFAR10"):
    results = get_batch_times(dataset_name)
    epoch_budgets = {}
    for idx in sorted(results.keys()):
        time_per_batch = results[idx]["train_mean"]
        epoch_budget = get_epoch_budget(dataset_name, time_per_batch)
        epoch_budgets[idx] = epoch_budget

    for i in range(8):
        for j in range(4):
            idx = i + 8*j
            v = epoch_budgets.get(idx, " - ")
            print(f"{str(v)[:8]: ^8}", end=" ")
        print()

    print("Epoch Budgets:")
    print(yaml.dump(epoch_budgets))

    fp = c.EPOCH_BUDGETS_2H_FILES[dataset_name]
    print(f"Save epoch budgets to file {fp} in order to use it for training.")

    # Save:
    # with open(constants.EPOCH_BUDGETS_2H_FILE, "w") as f:
    #     yaml.dump(epoch_budgets, f)


def get_epoch_budget(dataset_name, time_per_batch, nrof_hours=2):
    time_per_epoch = time_per_batch * get_nrof_batches_in_epoch(dataset_name)
    budget_seconds = nrof_hours * 60 * 60
    epoch_budget = int(math.floor(budget_seconds / time_per_epoch))
    return epoch_budget


def get_nrof_batches_in_epoch(dataset_name):
    ds_cls = getattr(data.datasets, dataset_name)
    ds = ds_cls()
    ds.prepare_data(val_proportion=0.)
    ds_len = len(ds.train)

    bs = c.BATCH_SIZE[dataset_name]
    return int(math.ceil(ds_len / bs))


def get_batch_times(dataset_name):
    bt_path = get_batch_times_path(dataset_name)
    if not os.path.exists(bt_path):
        print(f"File {bt_path} does not exist!")
        return {}

    with open(bt_path, "r") as f:
        lines = f.readlines()

    results = {}
    sep = c.SEPERATOR
    for line in lines:
        idx_str, result_dict_str, exception_str = line.strip().split(sep)
        if exception_str == "None":
            results[int(idx_str)] = eval(result_dict_str)

    return results

