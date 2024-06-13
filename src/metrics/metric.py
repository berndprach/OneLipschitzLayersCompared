
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


class Metric(ABC):
    print_as_percentage: bool = False
    aggregation: Callable[[Tensor], Tensor] = torch.mean

    @abstractmethod
    def __call__(self,
                 prediction_batch: Tensor,
                 label_batch: Tensor,  # labels not one-hot
                 ) -> Tensor:  # not aggregated
        raise NotImplementedError

    def aggregated(self, prediction_batch, label_batch) -> Tensor:
        return self.aggregation(self(prediction_batch, label_batch))

    # def get_name(self):
    #     if hasattr(self, "name"):
    #         return self.name
    #     return self.__class__.__name__
