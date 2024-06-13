
import unittest

import torch

from src.models.layers import ZeroChannelConcatenation


class TestZeroChannelConcatenation(unittest.TestCase):
    def test_4d(self):
        zcc = ZeroChannelConcatenation(12)
        input_tensor = torch.randn((4, 2, 4, 4))
        output_tensor = zcc(input_tensor)
        self.assertEqual(output_tensor.shape, (4, 12, 4, 4))

    def test_2d(self):
        zcc = ZeroChannelConcatenation(4)
        input_tensor = torch.randn((4, 12))
        output_tensor = zcc(input_tensor)
        self.assertEqual(output_tensor.shape, (4, 12))
