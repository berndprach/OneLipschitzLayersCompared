
import unittest

import torch
from torch import nn


class TestBasic(unittest.TestCase):
    def test_pixel_unshuffle(self):
        layer = nn.PixelUnshuffle(2)
        ones = torch.ones((4, 4))
        input_tensor = torch.stack([1*ones, 2*ones, 3*ones])[None]
        output_tensor = layer(input_tensor)

        goal_channels = sum((4*[i] for i in range(1, 4)), start=[])
        ones = torch.ones((2, 2))
        goal_output = torch.stack([ones*i for i in goal_channels])[None]

        self.assertTrue(torch.allclose(output_tensor, goal_output))
