
from .metric import Metric

import torch


class CertifiedRobustAccuracyFromScores(Metric):
    def __init__(self, maximal_perturbation, rescaling_factor=2. ** (1 / 2)):
        self.name = f"CRA{maximal_perturbation:.2f}"
        self.threshold = maximal_perturbation * rescaling_factor
        super().__init__()

    def __call__(self, score_batch, label_batch):
        label_batch_oh = torch.nn.functional.one_hot(
            label_batch.to(torch.int64),
            num_classes=score_batch.shape[-1]
        )
        penalized_scores = score_batch - self.threshold * label_batch_oh
        return torch.eq(penalized_scores.argmax(dim=1), label_batch).float()


CRAFromScores = CertifiedRobustAccuracyFromScores
CRA = CertifiedRobustAccuracyFromScores


def get_cra(margins: list[float], radius: float) -> float:
    correct = len([m for m in margins if m > radius])
    cra = correct / len(margins)
    return cra


STANDARD_CRA_RADII = [0., 36/255, 72/255, 108/255, 1.]
