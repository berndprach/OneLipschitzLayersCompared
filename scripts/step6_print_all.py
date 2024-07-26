import os

import yaml

from src.models.model_layer_combinations import all_combinations

from . import constants as c
from . import step3_hp_search as step3
from . import step5_test_set_evaluation as step5
from .util import convert_arguments_from_strings

model_sizes = {idx: comb[0] for idx, comb in enumerate(all_combinations)}
method_names = {idx: comb[1] for idx, comb in enumerate(all_combinations)}


@convert_arguments_from_strings
def main(dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    # indices = list(range(31))
    indices = [i + 8*j for i in range(8) for j in range(4)]  # by method

    epoch_budgets_2h = step3.get_epoch_budgets_2h(dataset_name)

    best_hps = step5.get_best_hps(dataset_name)
    best_lrs = {idx: hp["lr"] for idx, hp in best_hps.items()}
    best_wds = {idx: hp["wd"] for idx, hp in best_hps.items()}

    test_stats = get_test_stats(dataset_name)
    test_accs = {
        idx: f"{stats['Val_Accuracy']:.1%}"
        for idx, stats in test_stats.items()
    }
    test_cras = {
        idx: f"{stats['Val_CRA0.14']:.1%}"
        for idx, stats in test_stats.items()
    }

    table = Table(
        ("Index", indices),
        ("Model Size", model_sizes),
        ("Method Name", method_names),
        ("Epoch Budget", epoch_budgets_2h),
        ("Best LR", best_lrs),
        ("Best WD", best_wds),
        ("Accuracy", test_accs),
        ("Robust Acc", test_cras),
    )
    table.draw()


def get_test_stats(dataset_name):
    fp = c.get_test_results_path(dataset_name)
    if not os.path.exists(fp):
        print(f"Test results file ({fp}) does not exist.")
        return {}

    with open(fp, "r") as f:
        test_stats = yaml.load(f, Loader=yaml.SafeLoader)
    return test_stats


class Table:
    def __init__(self, key_column, *columns):
        self.keys = key_column[1]
        self.column_names = [key_column[0]] + [col[0] for col in columns]
        self.column_dicts = [col[1] for col in columns]

    def draw(self):
        print()
        self._draw_header()
        for key in self.keys:
            self._draw_row(key)
        print()

    def _draw_header(self):
        column_names = [f"{name[:12]: <12}" for name in self.column_names]
        header = " | ".join(column_names)
        print(header)
        print("-" * len(header))

    def _draw_row(self, key):
        values = [key] + [cd.get(key, " - ") for cd in self.column_dicts]
        entries = [f"{str(v)[:12]: ^12}" for v in values]
        row = " | ".join(entries)
        print(row)


# class Table:
#     def __init__(self, *columns):
#         self.column_names = [col[0] for col in columns]
#         self.column_values = [col[1] for col in columns]
#         self.w = 12  # entry width
#
#     def draw(self):
#         self._draw_header()
#         for row_values in zip(*self.column_values):
#             self._draw_row(row_values)
#
#     def _draw_header(self):
#         column_names = [f"{name[:self.w]: <{self.w}}"
#                         for name in self.column_names]
#         header = " | ".join(column_names)
#         print(header)
#         print("-" * len(header))
#
#     def _draw_row(self, row_values):
#         entries = [f"{str(v)[:self.w]: ^{self.w}}" for v in row_values]
#         row = " | ".join(entries)
#         print(row)
