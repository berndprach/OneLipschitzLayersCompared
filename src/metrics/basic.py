
from .metric import Metric

import torch
from torch import Tensor


class Accuracy(Metric):
    print_as_percentage = True

    def __call__(self, prediction_batch, label_batch):
        return torch.eq(prediction_batch.argmax(dim=1), label_batch).float()


class BatchVariance(Metric):
    """ Var over batch, summed over all features. """
    def __call__(self, prediction_batch, label_batch) -> Tensor:
        return prediction_batch.var(dim=0, correction=0).sum()


class Mean(Metric):
    """ Mean over batch, summed over all features. """
    def __call__(self, prediction_batch, label_batch) -> Tensor:
        return prediction_batch.mean(dim=0).sum()


class Margin(Metric):
    def __call__(self, score_batch, label_batch):
        if len(label_batch.shape) == 1:  #
            label_batch = torch.nn.functional.one_hot(
                label_batch.to(torch.int64),
                num_classes=score_batch.shape[-1]
            )
            label_batch = label_batch.to(score_batch.dtype)
        true_score = (score_batch * label_batch).sum(dim=-1)
        best_other = (score_batch - label_batch * 1e6).max(dim=-1)[0]
        return true_score - best_other


class Quantile(Metric):
    def __init__(self, metric, q, upper=True):
        self.metric = metric
        self.q = q if upper else 1 - q
        self.name = f"Q{q}" + metric.name

    def __call__(self, scores, labels):
        return torch.quantile(self.metric(scores, labels), self.q)
