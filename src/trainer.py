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


default_lf = SingleColumnLineFormatter(column_width=12)


@dataclass
class TrainerWithSideEffects:
    trainer: Trainer
    line_formatter: Callable[[dict], str] = default_lf
    timer: run_logging.Timer = run_logging.Timer()
    log: Callable[[str], None] = print

    post_epoch_callbacks: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self.timer.start()

    def process_epoch(self, data_iterator, epoch):
        train_statistics = self.trainer.train_epoch(data_iterator.train)
        val_statistics = self.trainer.evaluate(data_iterator.val)

        all_stats = {
            "Epoch": epoch,
            "AvgSec": self.timer.seconds_elapsed / epoch,
            **train_statistics,
            **val_statistics
        }
        self.log(self.line_formatter(all_stats))

        for cb in self.post_epoch_callbacks:
            cb()

    def train(self, data_iterator, nrof_epochs):
        self.start_timer()
        for epoch in range(1, nrof_epochs + 1):
            self.process_epoch(data_iterator, epoch)

    def start_timer(self):
        self.timer.start()
        print("Started training.")


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
