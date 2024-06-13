import math
import os

import yaml

from .constants import get_batch_times_path

BUDGET_SECONDS = 2 * 60 * 60  # 2 hours
NROF_BATCHES_IN_EPOCH = 50_000 // 256 + 1


def main(dataset_name: str = "CIFAR10"):
    results = get_results(dataset_name)
    epoch_budgets = {}
    for idx in sorted(results.keys()):
        time_per_batch = results[idx]["train_mean"]
        epoch_budget = get_epoch_budget(time_per_batch)
        epoch_budgets[idx] = epoch_budget

    # Save:
    # with open(constants.EPOCH_BUDGETS_2H_FILE, "w") as f:
    #     yaml.dump(epoch_budgets, f)

    print(yaml.dump(epoch_budgets))


def get_epoch_budget(time_per_batch, nrof_hours=2):
    time_per_epoch = time_per_batch * NROF_BATCHES_IN_EPOCH
    budget_seconds = nrof_hours * 60 * 60
    epoch_budget = int(math.floor(budget_seconds / time_per_epoch))
    return epoch_budget


def get_results(dataset_name):
    bt_path = get_batch_times_path(dataset_name)
    if not os.path.exists(bt_path):
        raise ValueError(f"File {bt_path} does not yet exist. "
                         f"First run step1_measure_batch_times!")

    results = {}
    with open(bt_path, "r") as f:
        for line in f:
            idx_str, result_dict_str, exception_str = line.strip().split("; ")
            if exception_str == "None":
                results[int(idx_str)] = eval(result_dict_str)
    return results
