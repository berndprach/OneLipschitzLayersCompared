import itertools

import torch

from . import layers
from . import simplified_conv_net

LINEAR_LAYERS = {
    "Cayley": layers.CayleyLinear,
    "BCOP": layers.BnBLinearBCOP,
}

# layers_names = [layer.__class__.__name__
# layers_names = [layer.__name__ for layer in layers.ALL_COMPARED_LAYERS]
# layer_names = ["StandardConv2d"] + layers.all_compared_layer_names
# method_names = ["AOL", "BCOP", "CPL", "Cayley", "LOT", "SLL", "SOC"]

layer_names = ["StandardConv2d"] + sorted(layers.COMPARED_LAYERS.keys())
model_names = ["XS", "S", "M", "L"]

all_combinations = list(itertools.product(model_names, layer_names))


def get_model_by_idx(idx: int) -> torch.nn.Sequential:
    size_name, method_name = all_combinations[idx]
    return get_model(size_name, method_name)


def get_model(size_name: str, method_name: str) -> torch.nn.Sequential:
    linear_cls = LINEAR_LAYERS.get(method_name, None)
    conv_cls = layers.COMPARED_LAYERS.get(method_name, layers.Conv2d)

    return simplified_conv_net.create_from_size(
        size_name=size_name,
        input_resolution=32,
        get_conv=conv_cls,
        get_linear=linear_cls
    )
