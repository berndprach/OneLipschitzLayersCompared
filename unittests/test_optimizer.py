
import unittest

import torch

from src import optimizer
from src.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):
    def test_one_circle_lr_scheduler(self):
        model = torch.nn.Linear(10, 10)
        base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            base_opt, max_lr=0.1, total_steps=100
        )
        opt = Optimizer(base_opt, scheduler)

        lrs = []
        for i in range(100):
            opt.zero_grad()
            opt.step()
            self.assertEqual(opt.learning_rate, scheduler.get_last_lr()[0])
            lrs.append(opt.learning_rate)

        self.assertAlmostEqual(max(lrs), 0.1, places=3)
        self.assertLess(lrs[0], 0.01)
        self.assertLess(lrs[-1], 1e-4)

        # import matplotlib.pyplot as plt
        # plt.plot(lrs)
        # plt.show()

    def test_one_cycle_sgd(self):
        pseudo_model = torch.nn.Linear(10, 10)
        sgd_hp = optimizer.OneCycleSGDHp(peak_lr=1., weight_decay=0.1)
        opt = optimizer.OneCycleSGD(
            pseudo_model.parameters(),
            total_steps=100,
            hp=sgd_hp,
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
