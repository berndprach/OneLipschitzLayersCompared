import os

import yaml

from src.models.model_layer_combinations import all_combinations

from . import constants as c
from . import step3_hp_search as step3
from . import step5_test_set_evaluation as step5
from .util import convert_arguments_from_strings


@convert_arguments_from_strings
def main(dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    model_sizes = {idx: comb[0] for idx, comb in enumerate(all_combinations)}
    method_names = {idx: comb[1] for idx, comb in enumerate(all_combinations)}

    epoch_budgets_2h = step3.get_epoch_budgets_2h(dataset_name)
    best_hps = step5.get_best_hps(dataset_name)
    best_lrs = {idx: best_hps[idx]["lr"] for idx in best_hps}
    best_wds = {idx: best_hps[idx]["wd"] for idx in best_hps}

    test_stats = get_test_stats(dataset_name)
    test_accs = {idx: stats["Accuracy"] for idx, stats in test_stats.items()}
    test_cras = {idx: stats["CRA36"] for idx, stats in test_stats.items()}

    table = Table(
        ("Index", list(range(31))),
        ("Model Size", model_sizes),
        ("Method Name", method_names),
        ("Epoch Budget", epoch_budgets_2h),
        ("Best LR", best_lrs),
        ("Best WD", best_wds),
        # ("Test Stats", test_stats),
        ("Accuracy", test_accs),
        ("Robust Accuracy", test_cras),
    )
    table.draw()


def get_test_stats(dataset_name):
    fp = c.get_test_results_path(dataset_name)
    if not os.path.exists(fp):
        print(f"Test results file ({fp}) does not exist.")
        return {}

    with open(fp, "r") as f:
        test_stats = yaml.load(f, Loader=yaml.SafeLoader)
        # for line in f:
        #     idx_str, stats_str = line.strip().split(c.SEPERATOR)
        #     idx = int(idx_str)
        #     stats = yaml.load(stats_str, Loader=yaml.SafeLoader)
        #     test_stats[idx] = stats
    return test_stats


class Table:
    def __init__(self, indices, *columns):
        self.keys = indices[1]
        id_dict = {idx: str(idx) for idx in self.keys}
        self.column_names = [indices[0]] + [col[0] for col in columns]
        self.column_dicts = [id_dict] + [col[1] for col in columns]

    def draw(self):
        self._draw_header()
        for key in self.keys:
            self._draw_row(key)

    def _draw_header(self):
        column_names = [f"{name[:12]: <12}" for name in self.column_names]
        header = " | ".join(column_names)
        print(header)
        print("-" * len(header))

    def _draw_row(self, key):
        values = [cd.get(key, " - ") for cd in self.column_dicts]
        entries = [f"{str(v)[:12]: ^12}" for v in values]
        row = " | ".join(entries)
        print(row)
