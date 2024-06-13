
from src.models.model_layer_combinations import all_combinations


def main():
    for idx, (model_name, layer_name) in enumerate(all_combinations):
        print(f"{idx: 3d}: {model_name:^12} & {layer_name:^15}")

