from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch
from typing import Optional

from tarp.services.evaluation import Extremum

class TrainerState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
        epochs: int = 10,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        gradient_clipping_threshold: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.history: list[dict[str, float]] = [{} for _ in range(epochs)]
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.gradient_clipping_threshold = gradient_clipping_threshold

        self.epoch = 0
        self.stop_training = False


class TrainerContext:
    def __init__(self, state: TrainerState):
        self.state = state

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def model(self) -> nn.Module:
        return self.state.model

    @property
    def optimizer(self) -> Optimizer:
        return self.state.optimizer

    @property
    def scheduler(self) -> Optional[LRScheduler]:
        return self.state.scheduler

    @property
    def scaler(self) -> Optional[torch.amp.GradScaler]:
        return self.state.scaler

    def request_stop(self):
        self.state.stop_training = True

    def should_stop(self) -> bool:
        return self.state.stop_training

    def increment_epoch(self):
        self.state.epoch += 1

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def accumulation_steps(self) -> int:
        return self.state.accumulation_steps

    @property
    def use_amp(self) -> bool:
        return self.state.use_amp

    @property
    def gradient_clipping_threshold(self) -> float:
        return self.state.gradient_clipping_threshold

    @property
    def epochs(self) -> int:
        return self.state.epochs

    def record_current_history(self, metrics: dict[str, float]):
        self.state.history[self.epoch].update(metrics)

    @property
    def current_metrics(self) -> dict[str, float]:
        return self.state.history[self.epoch]