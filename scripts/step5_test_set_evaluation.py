
import torch
import yaml

from src import data, metrics, run_logging
from src.run_logging import Timer
from src.trainer import Trainer

from . import constants
from .step3_hp_search import get_epoch_budget, get_optimizer, get_metrics
from .step1_measure_batch_time import get_model_from_idx
from .util import convert_arguments_from_strings

SEP = ";"


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    model = get_model_from_idx(idx)
    data_preprocessor = get_data_preprocessor(dataset_name)

    best_hps = get_best_hps(idx, dataset_name)

    epochs = get_epoch_budget(idx, dataset_name, nrof_hours)
    print(f"Training model {idx} for {epochs} epochs.")

    loss_function = metrics.OffsetXent(
        offset=constants.LOSS_OFFSET,
        temperature=constants.LOSS_TEMPERATURE
    )
    optimizer = get_optimizer(model, best_hps["lr"], best_hps["wd"], epochs)
    trainer = Trainer(
        model, loss_function, optimizer, get_metrics(loss_function)
    )
    train_no_val(trainer, data_preprocessor, epochs)

    final_val_stats = trainer.evaluate(data_preprocessor.test)
    with open(constants.TEST_RESULTS_FILE, "a") as f:
        f.write(f"{idx}{SEP} {final_val_stats}\n")


def get_best_hps(idx, dataset_name):
    best_hp_fp = constants.BEST_HP_FILES[dataset_name]
    with open(best_hp_fp, "r") as f:
        best_hps = yaml.load(f, Loader=yaml.SafeLoader)
    return best_hps[idx]


def get_data_preprocessor(dataset_name):
    dataset = getattr(data.datasets, dataset_name)()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return data.get_dp(
        dataset, constants.BATCH_SIZE, device, val_proportion=0.1
    )


def train_no_val(trainer, data_preprocessor, epochs):
    timer = Timer()
    line_formatter = run_logging.SingleColumnLineFormatter()
    timer.start()

    for epoch in range(1, epochs+1):
        train_stats = trainer.train_epoch(data_preprocessor.train)

        all_stats = {
            "Epoch": epoch,
            "AvgSec": timer.seconds_elapsed / epoch,
            **train_stats,
        }
        print(line_formatter(all_stats))

