from dataclasses import dataclass, field
from typing import Callable, List, Protocol

import torch

from src import metrics, run_logging
from src.run_logging import BatchTracker, Timer
from src.run_logging.line_formatter import SingleColumnLineFormatter


class Optimizer(Protocol):
    def zero_grad(self):
        ...

    def step(self):
        ...


@dataclass
class Trainer:
    model: torch.nn.Module
    loss_function: Callable
    optimizer: Optimizer
    tracked_metrics: List[metrics.Metric]

    def train_epoch(self, train_loader, prefix="Train_"):
        batch_tracker = BatchTracker(self.tracked_metrics)
        self.model.train()

        for batch in train_loader:
            self.train_batch(*batch, batch_tracker)

        train_metrics = batch_tracker.get_average_results(prefix)
        return train_metrics

    def train_batch(self, x_batch, y_batch, batch_tracker):
        # Forward pass
        outputs = self.model(x_batch)
        loss = self.loss_function(outputs, y_batch)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Evaluate the metrics
        batch_tracker.update(outputs, y_batch)

    def evaluate(self, val_loader, prefix="Val_"):
        batch_tracker = BatchTracker(self.tracked_metrics)
        self.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                self.evaluate_batch(*batch, batch_tracker)

        val_metrics = batch_tracker.get_average_results(prefix)
        return val_metrics

    def evaluate_batch(self, x_batch, y_batch, batch_tracker):
        outputs = self.model(x_batch)
        batch_tracker.update(outputs, y_batch)


default_formatter = run_logging.DoubleColumnsLineFormatter(6)


def train_model(trainer, dp, epochs, line_formatter=default_formatter):
    timer = Timer()
    timer.start()

    for epoch in range(1, epochs+1):
        train_stats = trainer.train_epoch(dp.train)
        val_stats = trainer.evaluate(dp.val)

        all_stats = {
            "Epoch": epoch,
            "AvgSec": timer.seconds_elapsed / epoch,
            **train_stats,
            **val_stats
        }
        print(line_formatter(all_stats))
        trainer.optimizer.scheduler_step()
