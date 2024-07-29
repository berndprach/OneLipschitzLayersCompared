from scripts.util import convert_arguments_from_strings
from src import data
from src.data.get_data_preprocessor import get_augmented_dp
from src.evaluations.batch_times import get_mean_batch_seconds
from src.models.model_layer_combinations import get_model_by_idx

from . import constants as c
from .constants import SEPERATOR

NROF_BATCHES = 100


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10"):
    model = get_model_by_idx(idx, c.NROF_BLOCKS[dataset_name])
    data_preprocessor = get_augmented_dp(dataset_name, c.BATCH_SIZE, 0.)
    results_fp = c.get_batch_times_path(dataset_name)

    try:
        batch_times = evaluate_batch_times(model, data_preprocessor)
        result_str = str(batch_times)
        save_to_file(results_fp, idx, result_str, exception_str="None")
    except Exception as e:
        print(f"An exception occurred: {e}")
        exception_str = str(e).replace(SEPERATOR, ",")
        save_to_file(results_fp, idx, "None", exception_str)
        raise e


def save_to_file(results_fp, idx, result_str, exception_str="None"):
    with open(results_fp, "a") as f:
        f.write(f"{idx}{SEPERATOR}{result_str}{SEPERATOR}{exception_str}\n")
    print(f"Saved result to {results_fp}.")


def evaluate_batch_times(model, data_preprocessor, nrof_batches=NROF_BATCHES):
    return get_mean_batch_seconds(
        model,
        train_loader=data_preprocessor.train,
        test_loader=data_preprocessor.test,
        nrof_batches=nrof_batches,
    )
