
import unittest

import torch

from src import optimizer
from src.optimizer import OneCycleSGD


class TestOptimizer(unittest.TestCase):
    def test_one_circle_lr_scheduler(self):
        model = torch.nn.Linear(10, 10)
        opt = OneCycleSGD(model.parameters(), total_steps=100, peak_lr=0.1)

        lrs = []
        for i in range(100):
            opt.zero_grad()
            opt.step()
            opt.scheduler_step()
            lrs.append(opt.learning_rate)

        self.assertAlmostEqual(max(lrs), 0.1, places=3)

        self.assertAlmostEqual(lrs[0], 0.1/25, places=3)
        self.assertAlmostEqual(lrs[30], 0.1, places=3)
        self.assertAlmostEqual(lrs[-1], 0.1/1e4, places=3)

        # import matplotlib.pyplot as plt
        # plt.title("Learning Rate With OneCycleSGD")
        # plt.plot(lrs)
        # plt.show()

    def test_one_cycle_sgd(self):
        pseudo_model = torch.nn.Linear(10, 10)
        opt = optimizer.OneCycleSGD(
            pseudo_model.parameters(),
            total_steps=100,
            peak_lr=1.,
            weight_decay=0.1,
        )
        lrs = []
        for _ in range(100):
            lrs.append(opt.learning_rate)
            opt.step()
            opt.scheduler_step()

        self.assertLess(lrs[0], 0.1)
        self.assertAlmostEqual(max(lrs), 1., places=3)
        self.assertLess(lrs[-1], 1e-4)
        self.assertTrue(all(0. <= lr <= 1. for lr in lrs))
