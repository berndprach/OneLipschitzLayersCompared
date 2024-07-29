import itertools

import torch
from torch import nn

from . import layers
from . import simplified_conv_net

LINEAR_LAYERS = {
    "Cayley": layers.CayleyLinear,
    "BCOP": layers.BnBLinearBCOP,
}

layer_names = ["StandardConv2d"] + sorted(layers.COMPARED_LAYERS.keys())
model_names = ["XS", "S", "M", "L"]

all_combinations = list(itertools.product(model_names, layer_names))


def get_model_by_idx(idx: int, nrof_blocks=5) -> nn.Sequential:
    size_name, method_name = all_combinations[idx]
    print(f"Chosen combination {idx}:", end=" ")
    print(f"model {size_name} with layer {method_name}.")
    return get_model(size_name, method_name, nrof_blocks=nrof_blocks)


def get_model(size_name: str, method_name: str, **kwargs) -> nn.Sequential:
    conv_cls = layers.COMPARED_LAYERS.get(method_name, layers.Conv2d)
    linear_cls = LINEAR_LAYERS.get(method_name, None)

    model = simplified_conv_net.create_from_size(
        size_name=size_name,
        get_conv=conv_cls,
        get_linear=linear_cls,
        **kwargs,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

