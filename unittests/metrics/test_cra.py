
import unittest

import torch

from src.metrics import CRAFromScores


class TestCRA(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor([
            [1., 0.],
            [2., 0.],
            [0., 1.],
            [0., 2.],
            [1.1, 0.9],
        ])
        self.labels = torch.zeros(5, dtype=torch.int64)

    def test_cra_without_offset(self):
        cra_metric = CRAFromScores(0.)
        cra = cra_metric(self.scores, self.labels)
        self.assertEqual(list(cra), [1., 1., 0., 0., 1.])

    def test_cra_with_offset(self):
        cra_metric = CRAFromScores(1.)
        cra = cra_metric(self.scores, self.labels)
        self.assertEqual(list(cra), [0., 1., 0., 0., 0.])

    def test_rescaling_factor(self):
        cra_metric = CRAFromScores(1., rescaling_factor=1/2)
        cra = cra_metric(self.scores, self.labels)
        self.assertEqual(list(cra), [1., 1., 0., 0., 0.])


