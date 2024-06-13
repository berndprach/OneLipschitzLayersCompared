
from .metric import Metric

import torch


class OffsetCrossEntropyFromScores(Metric):
    def __init__(self, offset=0., temperature=1., **kwargs):
        super().__init__()
        self.offset = offset
        self.temperature = temperature
        self.name = f"OX({offset:.2g}, {temperature:.2g})"
        self.std_xent = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, score_batch, label_batch):
        if len(label_batch.shape) == 1:  #
            label_batch = torch.nn.functional.one_hot(
                label_batch.to(torch.int64),
                num_classes=score_batch.shape[-1]
            )
            label_batch = label_batch.to(score_batch.dtype)
        offset_scores = score_batch - self.offset * label_batch
        offset_scores /= self.temperature
        return self.std_xent(offset_scores, label_batch) * self.temperature


OffsetXent = OffsetCrossEntropyFromScores


