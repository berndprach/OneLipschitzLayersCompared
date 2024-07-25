
import torch

from dataclasses import dataclass

from torch.optim.lr_scheduler import OneCycleLR

from src.hyperparameters import Hyperparameters


@dataclass
class OneCycleSGDHp(Hyperparameters):
    peak_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.
    nesterov: bool = True


class OneCycleSGD:
    def __init__(self, params, total_steps: int, **kwargs):
        hp = OneCycleSGDHp(**kwargs)
        self.sgd = torch.optim.SGD(
            params,
            lr=0.,
            momentum=hp.momentum,
            nesterov=hp.nesterov,
            weight_decay=hp.weight_decay,
        )
        self.scheduler = OneCycleLR(
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
