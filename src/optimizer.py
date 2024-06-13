
import torch

from torch.optim.lr_scheduler import LRScheduler

from dataclasses import dataclass

from src.hyperparameters import HP


@dataclass
class Optimizer:
    base_optimizer: torch.optim.Optimizer
    scheduler: LRScheduler

    @property
    def learning_rate(self):
        return self.base_optimizer.param_groups[0]["lr"]

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        self.base_optimizer.step()
        self.scheduler.step()


# @dataclass
# class SGDHp(HP):
#     momentum: float = 0.9
#     nesterov: bool = True
#     weight_decay: float = 0.

@dataclass
class OneCycleSGDHp(HP):
    peak_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.
    nesterov: bool = True


class OneCycleSGD:
    def __init__(self, params, total_steps: int, hp: OneCycleSGDHp):
        self.sgd = torch.optim.SGD(
            params,
            lr=0.,
            momentum=hp.momentum,
            nesterov=hp.nesterov,
            weight_decay=hp.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.sgd, max_lr=hp.peak_lr, total_steps=total_steps
        )

    @property
    def learning_rate(self):
        return get_lr(self.sgd)

    def step(self):
        self.sgd.step()

    def zero_grad(self):
        self.sgd.zero_grad()

    def scheduler_step(self):
        self.scheduler.step()


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
