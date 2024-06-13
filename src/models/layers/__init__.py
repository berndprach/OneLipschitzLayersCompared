from functools import partial

import torch

from .lipschitz.aol import AOLConv2d, AOLConv2dDirac, AOLConv2dOrth
from .lipschitz.bnb import BnBLinear, BnBLinearBCOP
from .lipschitz.bcop import BCOP
from .lipschitz.cayley import CayleyConv, CayleyLinear
from .lipschitz.cpl import CPLConv2d, CPLConv2d10k
from .lipschitz.lot import LOT, LOT2t
from .lipschitz.sll import SLLConv2d
from .lipschitz.soc import SOC

from .lipschitz.spectral_normal_control import *

from .activations import MaxMin, Abs, Identity
from .basic import ZeroChannelConcatenation

Conv2d = partial(torch.nn.Conv2d, padding='same')

COMPARED_LAYERS = {
    "AOL": AOLConv2d,
    "BCOP": BCOP,
    "CPL": CPLConv2d,
    "Cayley": CayleyConv,
    "LOT": LOT,
    "SLL": SLLConv2d,
    "SOC": SOC,
}


# ALL_COMPARED_LAYERS = [
#     AOLConv2d,
#     BCOP,
#     CPLConv2d,
#     CayleyConv,
#     LOT,
#     SOC,
#     SLLConv2d,
# ]
# all_compared_layer_names = [layer.__name__ for layer in ALL_COMPARED_LAYERS]
