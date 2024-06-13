import math
import yaml

from . import constants
from .constants import get_batch_times_path

BUDGET_SECONDS = 2 * 60 * 60  # 2 hours
NROF_BATCHES_IN_EPOCH = 50_000 // 256 + 1


def main(dataset_name: str = "CIFAR10"):
    results = get_results(dataset_name)
    best_hps = get_best_hps(results)
    print(yaml.dump(best_hps))


def get_epoch_budget(time_per_batch, nrof_hours=2):
    time_per_epoch = time_per_batch * NROF_BATCHES_IN_EPOCH
    budget_seconds = nrof_hours * 60 * 60
    epoch_budget = int(math.floor(budget_seconds / time_per_epoch))
    return epoch_budget


def get_results(dataset_name):
    results = {}
    with open(constants.HP_SEARCH_RESULTS_FILE, "r") as f:
        for line in f:
            idx_str, hp_dict_str, final_stats_str = line.strip().split("; ")
            idx = int(idx_str)
            if idx not in results:
                results[idx] = []

            results[idx].append((eval(hp_dict_str), eval(final_stats_str)))

    for key in sorted(results.keys()):
        print(f"Found {len(results[key])} results for index {key}.")

    return results


def get_best_hps(results):
    best_hps = {}
    for idx in results:
        best_hp = None
        best_val_acc = 0
        for hp, stats in results[idx]:
            val_cra = stats["Val_CRA0.14"]
            if val_cra > best_val_acc:
                best_hp = hp
                best_val_acc = val_cra

        best_hps[idx] = best_hp

    return best_hps
