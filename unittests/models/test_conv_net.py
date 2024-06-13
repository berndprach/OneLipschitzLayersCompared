import unittest

import torch

from src import models
from src.models import layers


MIL = 1_000_000
# BASE_WIDTHS = {"XS": 16, "S": 32, "M": 64, "L": 128}
W_TO_LB = {16: 1*MIL, 32: 4*MIL, 64: 16*MIL, 128: 64*MIL}


class TestSimplifiedConvNet(unittest.TestCase):
    def test_output_shape(self):
        cn = models.simplified_conv_net.create(nrof_classes=100)
        input_tensor = torch.randn((4, 3, 32, 32))
        output_tensor = cn(input_tensor)
        self.assertEqual(output_tensor.shape, (4, 100))

    def test_number_of_layers(self):
        cn = models.simplified_conv_net.create(
            nrof_layers_per_block=6, nrof_blocks=3
        )
        self.assertEqual(_count(torch.nn.Conv2d, cn), 6*3 + 2)
        self.assertEqual(_count(layers.MaxMin, cn), 6*3 + 1)

    def test_number_of_parameters(self):
        cn = models.simplified_conv_net.create(base_width=16)
        self.assertTrue(1*MIL < _count_parameters(cn) <= 2*MIL)

        for w in [16, 32, 64, 128]:
            cn = models.simplified_conv_net.create(base_width=w)
            p_lb = W_TO_LB[w]
            self.assertTrue(p_lb < _count_parameters(cn) < 2*p_lb)

    def test_different_resolutions(self):
        cn = models.simplified_conv_net.create_from_size("XS", 64)
        self.assertEqual(_count(torch.nn.Conv2d, cn), 6*5 + 2)
        self.assertTrue(1*MIL < _count_parameters(cn) <= 2*MIL)

        cn = models.simplified_conv_net.create_from_size("M", 256)
        self.assertEqual(_count(torch.nn.Conv2d, cn), 8*5 + 2)
        self.assertTrue(16*MIL < _count_parameters(cn) <= 32*MIL)


def _count(module_type, model):
    return sum(1 for m in model.modules() if isinstance(m, module_type))


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters())
