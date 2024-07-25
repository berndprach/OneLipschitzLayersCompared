
import yaml

from src import run_logging
from src.data.get_data_preprocessor import get_data_preprocessor
from src.run_logging import Timer
from src.models.model_layer_combinations import get_model_by_idx

from . import constants as c
from . import step3_hp_search as step3
from .util import convert_arguments_from_strings


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10", nrof_hours: float = 2):
    model = get_model_by_idx(idx)
    data_preprocessor = get_data_preprocessor(dataset_name, c.BATCH_SIZE, 0.)
    epochs = step3.get_epoch_budget(idx, dataset_name, nrof_hours)
    best_hps = get_best_hps(dataset_name)[idx]

    print(f"Training model {idx} for {epochs} epochs with {best_hps}")

    trainer = step3.get_trainer(epochs, model, best_hps["lr"], best_hps["wd"])
    train_no_val(trainer, data_preprocessor, epochs)

    final_val_stats = trainer.evaluate(data_preprocessor.test)

    fp = c.get_test_results_path(dataset_name)
    save_to(fp, final_val_stats, idx)


def save_to(fp, final_val_stats, idx):
    with open(fp, "a") as f:
        # f.write(f"{idx}{c.SEPERATOR}{final_val_stats}\n")
        yaml.dump({idx: final_val_stats}, f)
    print(f"Appended test set performance for file {fp}.")


def get_best_hps(dataset_name):
    best_hp_fp = c.BEST_HP_FILES[dataset_name]
    print(f"Loading best hyperparameters from {best_hp_fp}.")
    with open(best_hp_fp, "r") as f:
        best_hps = yaml.load(f, Loader=yaml.SafeLoader)
    return best_hps


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
        trainer.optimizer.scheduler_step()

