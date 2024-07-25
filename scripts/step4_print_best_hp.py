import math

from dataclasses import dataclass
from typing import Dict, Any, List

import yaml

from . import constants as c

BUDGET_SECONDS = 2 * 60 * 60  # 2 hours
NROF_BATCHES_IN_EPOCH = 50_000 // 256 + 1


def main(dataset_name: str = "CIFAR10"):
    results = get_results(dataset_name)
    best_hps = get_best_hps(results)

    print("Best Hyperparameters:")
    print(yaml.dump(best_hps))

    fp = c.BEST_HP_FILES[dataset_name]
    print(f"Save epoch budgets to file {fp} in order to use it for training.")


def get_epoch_budget(time_per_batch, nrof_hours=2):
    time_per_epoch = time_per_batch * NROF_BATCHES_IN_EPOCH
    budget_seconds = nrof_hours * 60 * 60
    epoch_budget = int(math.floor(budget_seconds / time_per_epoch))
    return epoch_budget


def get_results(dataset_name: str):
    fp = c.get_hp_search_results_path(dataset_name)
    with open(fp, "r") as f:
        results = get_results_from_file(f)

    for key in sorted(results.keys()):
        print(f"Found {len(results[key])} results for index {key}.")

    return results


Hps = Dict[str, Any]
ValStats = Dict[str, float]


# Result = Tuple[Hps, ValStats]

@dataclass
class Result:
    hps: Hps
    val_stats: ValStats


ResultsByIndex = Dict[int, List[Result]]


def get_results_from_file(f) -> ResultsByIndex:
    results = {}
    for line in f:
        idx_str, hp_dict_str, final_stats_str = line.strip().split("; ")
        idx = int(idx_str)
        if idx not in results:
            results[idx] = []

        result = Result(eval(hp_dict_str), eval(final_stats_str))
        results[idx].append(result)
        # results[idx].append((eval(hp_dict_str), eval(final_stats_str)))
    return results


def get_best_hps(results_by_index: ResultsByIndex) -> Dict[int, Hps]:
    best_hps = {
        idx: get_best_hps_for_idx(result_list)
        for idx, result_list in results_by_index.items()
    }
    return best_hps


def get_best_hps_for_idx(results: List[Result]) -> Hps:
    best_list_idx = max(
        range(len(results)),
        key=lambda i: results[i].val_stats["Val_CRA0.14"]
    )
    return results[best_list_idx].hps
