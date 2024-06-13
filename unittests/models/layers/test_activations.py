import unittest

import torch

from src.models import layers


class TestActivations(unittest.TestCase):
    def test_abs(self):
        layer = layers.Abs()
        input_tensor = torch.Tensor([[-1, 0, 1], [-2, 0, 2]])
        output_tensor = layer(input_tensor)
        goal_output = torch.Tensor([[1, 0, 1], [2, 0, 2]])
        self.assertTrue(torch.allclose(output_tensor, goal_output))

    def test_max_min(self):
        layer = layers.MaxMin()
        input_tensor = torch.Tensor([[-1, 0, 1, -2], [-2, 0, 2, 2]])
        output_tensor = layer(input_tensor)
        goal_output = torch.Tensor([[0, -1, 1, -2], [0, -2, 2, 2]])
        self.assertTrue(torch.allclose(output_tensor, goal_output))

