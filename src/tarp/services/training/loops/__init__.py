from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Optional, Callable

from tarp.services.training.context import TrainerContext

class Loop(ABC):
    def __init__(
        self,
        context: TrainerContext,
        iteration: Callable[
            [dict[str, Tensor]], tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        ],
        evaluation: Callable[
            [list[Tensor], list[Tensor]], dict[str, float]
        ] = lambda prediction, expected: {},
    ):
        """
        Base class for training/evaluation loops.

        :param context: TrainerContext providing access to trainer state.
        :param iteration: Function to perform a single iteration (training/validation step).
        :param evaluation: Function to compute metrics given predictions and expected values.
        """
        self.context = context
        self.iteration = iteration
        self.evaluation = evaluation

    @abstractmethod
    def run(self, epoch: int, dataloader: DataLoader) -> dict[str, float]:
        raise NotImplementedError