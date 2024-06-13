import torch

from src import data

from scripts.util import convert_arguments_from_strings
from src.evaluations.batch_times import evaluate_all_model_time_statistics
from src.models.model_layer_combinations import all_combinations, get_model

from .constants import get_batch_times_path

SEPERATOR = ";"
NROF_BATCHES = 100
BATCH_SIZE = 256


@convert_arguments_from_strings
def main(idx: int, dataset_name: str = "CIFAR10"):
    model = get_model_from_idx(idx)
    data_preprocessor = get_data_preprocessor(dataset_name)
    results_fp = get_batch_times_path(dataset_name)

    try:
        save_results_to_file(model, data_preprocessor, idx, results_fp)
    except Exception as e:
        print(f"An error occurred: {e}")
        save_exception_to_file(e, idx, results_fp)
        raise e


def save_results_to_file(model, data_preprocessor, idx, results_fp):
    result_dict = evaluate_batch_time(model, data_preprocessor, NROF_BATCHES)
    with open(results_fp, "a") as f:
        f.write(f"{idx}{SEPERATOR} {result_dict}{SEPERATOR} {None}\n")


def save_exception_to_file(exception, idx, results_fp):
    exception_str = str(exception).replace(SEPERATOR, ",")
    with open(results_fp, "a") as f:
        f.write(f"{idx}{SEPERATOR} {None}{SEPERATOR} {exception_str}\n")


def get_model_from_idx(idx):
    model_name, layer_name = all_combinations[idx]
    print(f"Chosen combination {idx}: "
          f"model {model_name} and layer {layer_name}.\n")
    return get_model(model_name, layer_name)


def get_data_preprocessor(dataset_name):
    dataset = getattr(data.datasets, dataset_name)()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return data.get_dp(dataset, BATCH_SIZE, device, val_proportion=0.)


def evaluate_batch_time(model, data_preprocessor, nrof_batches):
    return evaluate_all_model_time_statistics(
        model,
        train_loader=data_preprocessor.train,
        test_loader=data_preprocessor.test,
        nrof_batches=nrof_batches,
    )
