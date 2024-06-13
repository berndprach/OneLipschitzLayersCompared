import unittest

from src.models import layers, model_layer_combinations


MIL = 1_000_000
W_TO_LB = {16: 1*MIL, 32: 4*MIL, 64: 16*MIL, 128: 64*MIL}


class TestCombinations(unittest.TestCase):
    def test_get_by_idx(self):
        aol_s = model_layer_combinations.get_model_by_idx(1)
        self.assertEqual(_count(layers.AOLConv2d, aol_s), 5*5 + 2)

    def test_get_model_with_linear(self):
        cayley_m = model_layer_combinations.get_model("M", "Cayley")
        self.assertEqual(_count(layers.CayleyLinear, cayley_m), 1)
        self.assertEqual(_count(layers.CayleyConv, cayley_m), 5*5 + 1)


def _count(module_type, model):
    return sum(1 for m in model.modules() if isinstance(m, module_type))


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters())
