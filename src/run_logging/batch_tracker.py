from typing import List

import torch

from src.metrics.basic import Metric


class BatchTracker:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.batch_results = None

        self.reset()

    def reset(self):
        self.batch_results = {get_name(metric): [] for metric in self.metrics}

    def update(self, output_batch, label_batch):
        with torch.no_grad():
            for metric in self.metrics:
                # batch_aggregated = metric.aggregated(output_batch, label_batch)
                # # self.batch_results[metric.get_name()].append(batch_aggregated)
                # self.batch_results[get_name(metric)].append(batch_aggregated)
                statistics_batch = metric(output_batch, label_batch)
                # print(statistics_batch)
                if len(statistics_batch.shape) == 0:
                    statistics_batch = statistics_batch[None]
                self.batch_results[get_name(metric)].extend(statistics_batch)

    def get_average_results(self, prefix=""):
        return {f"{prefix}{k}": (sum(v) / len(v)).item()
                for k, v in self.batch_results.items()}


def get_name(metric):
    if hasattr(metric, "name"):
        return metric.name
    return metric.__class__.__name__
