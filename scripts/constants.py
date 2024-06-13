import os

BATCH_SIZE = 256

ROB_EPS = 36 / 255
LOSS_OFFSET = 2 * 2**0.5 * ROB_EPS
LOSS_TEMPERATURE = 1/4


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_TIMES_FILE = os.path.join(OUTPUT_DIR, "batch_time_results.csv")
EPOCH_BUDGETS_2H_FILE = os.path.join(OUTPUT_DIR, "epoch_budgets_2h.yaml")

HP_SEARCH_RESULTS_FILE = os.path.join(OUTPUT_DIR, "hp_search_results.csv")
TEST_RESULTS_FILE = os.path.join(OUTPUT_DIR, "test_results.csv")


DATA_DIR = "data"
EPOCH_BUDGETS_2H_DIR = os.path.join(DATA_DIR, "epoch_budgets_2h")
EPOCH_BUDGETS_2H_FILES = {
    "CIFAR10": os.path.join(EPOCH_BUDGETS_2H_DIR, "cifar10.yaml"),
}
BEST_HP_DIR = os.path.join(DATA_DIR, "best_hps")
BEST_HP_FILES = {
    "CIFAR10": os.path.join(BEST_HP_DIR, "cifar10.yaml"),
}


def get_batch_times_path(dataset_name: str):
    return os.path.join(OUTPUT_DIR, f"batch_times_{dataset_name}.csv")
