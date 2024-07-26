import os
from enum import Enum

from src import metrics

BATCH_SIZE = 256
SEPERATOR = "; "

ROB_EPS = 36 / 255
LOSS_OFFSET = 2 * 2 ** 0.5 * ROB_EPS
LOSS_TEMPERATURE = 1 / 4

METRICS = [
    metrics.Accuracy(),
    metrics.BatchVariance(),
    metrics.Margin(),
    metrics.CRA(1 * 36 / 255),
    metrics.CRA(2 * 36 / 255),
    metrics.CRA(3 * 36 / 255),
    metrics.CRA(1),
]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BATCH_TIMES_FILE = os.path.join(OUTPUT_DIR, "batch_time_results.csv")
# EPOCH_BUDGETS_2H_FILE = os.path.join(OUTPUT_DIR, "epoch_budgets_2h.yaml")

# HP_SEARCH_RESULTS_FILE = os.path.join(OUTPUT_DIR, "hp_search_results.csv")
# TEST_RESULTS_FILE = os.path.join(OUTPUT_DIR, "test_results.csv")


DATA_DIR = "data"
EPOCH_BUDGETS_2H_FILES = {
    "CIFAR10": os.path.join(DATA_DIR, "epoch_budgets_2h", "cifar10.yaml"),
}
BEST_HP_FILES = {
    "CIFAR10": os.path.join(DATA_DIR, "best_hps", "cifar10.yaml"),
}


def get_batch_times_path(dataset_name: str):
    # return os.path.join(OUTPUT_DIR, f"batch_times_{dataset_name}.csv")
    ds_path = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)
    return os.path.join(OUTPUT_DIR, dataset_name, f"batch_times.csv")


def get_hp_search_results_path(dataset_name: str):
    return os.path.join(OUTPUT_DIR, dataset_name, "hp_search_results.csv")


def get_test_results_path(dataset_name: str):
    return os.path.join(OUTPUT_DIR, dataset_name, "test_results.csv")
