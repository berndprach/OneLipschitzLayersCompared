
import unittest

import torch
from torch import nn

from src.metrics import OffsetCrossEntropyFromScores, OffsetXent


class TestOffsetXent(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor([
            [1., 0.],
            [2., 0.],
            [0., 1.],
            [0., 2.],
            [1.1, 0.9],
        ])
        self.labels = torch.zeros(5, dtype=torch.int64)

    def test_ox_equals_standard_cross_entropy(self):
        ox_metric = OffsetXent(reduction="none")
        ox = ox_metric(self.scores, self.labels)
        std_xent_loss = nn.CrossEntropyLoss(reduction="none")
        std_xent = std_xent_loss(self.scores, self.labels)
        self.assertEqual(list(ox), list(std_xent))

    def test_gradient_norm_in_01(self):
        ox_metric = OffsetCrossEntropyFromScores()
        self.scores.requires_grad = True
        ox = ox_metric(self.scores, self.labels)
        ox.sum().backward()
        all_small = torch.all(torch.less(torch.abs(self.scores.grad), 1))
        self.assertTrue(all_small)

    def test_gradient_norm_one_with_large_offset(self):
        ox_metric = OffsetCrossEntropyFromScores(offset=100., reduction="none")
        self.scores.requires_grad = True
        ox = ox_metric(self.scores, self.labels)
        ox.sum().backward()
        all_one = torch.all(torch.eq(torch.abs(self.scores.grad), 1))
        self.assertTrue(all_one)

    def test_gradient_zero_with_large_negative_offset(self):
        ox_metric = OffsetCrossEntropyFromScores(offset=-100.)
        self.scores.requires_grad = True
        ox = ox_metric(self.scores, self.labels)
        ox.sum().backward()
        self.assertTrue(
            torch.all(torch.less(torch.abs(self.scores.grad), 1e-5))
        )

    def test_tiny_temperature_makes_hinge(self):
        ox_metric = OffsetXent(temperature=1e-5, reduction="none")
        ox = ox_metric(self.scores, self.labels)
        margins = self.scores[:, 0] - self.scores[:, 1]
        hinge = torch.max((-1) * margins, torch.zeros(1))
        self.assertEqual(list(ox), list(hinge))



