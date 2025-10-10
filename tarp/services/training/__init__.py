from abc import ABC, abstractmethod
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Optional
from tqdm import tqdm


from tarp.services.evaluation import Extremeum
from tarp.services.loggers.colored import ColoredLogger


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        use_amp: bool = True,
        monitor_metric: str = "validation_loss",
        monitor_mode: Extremeum = Extremeum.MIN,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        self.scaler = torch.amp.GradScaler(self.device, enabled=use_amp)
        self.history: list[dict[str, float]] = [{} for _ in range(self.epochs)]

    @abstractmethod
    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Perform a single training step.

        :param batch: A batch of data from the DataLoader.
        :return Tensor: The computed loss for the batch.
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Perform a single validation step.

        :param batch: A batch of data from the DataLoader.
        :return tuple[Tensor, Optional[Tensor], Optional[Tensor]]: The computed loss for the batch, preds, and ground truths.
        """
        raise NotImplementedError

    def compute_metrics(
        self, prediction: list[Tensor], expected: list[Tensor]
    ) -> dict[str, float]:
        """
        Compute metrics given logits and true labels.

        :param list[Tensor] prediction: List of model predictions (logits).
        :param list[Tensor] expected: List of true labels.
        :return dict[str, float]: A dictionary of computed metrics.
        """
        return {}

    def _training_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        loop = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            unit="batch",
            colour="green",
        )
        for batch in loop:
            self.optimizer.zero_grad()
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                loss = self.training_step(batch)
            self.scaler.scale(loss).backward()

            # Unsacle gradients and perform gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = total_loss / len(self.train_dataloader)
        return average_loss

    def _validation_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_expected, all_predictions = [], []
        loop = tqdm(
            self.valid_dataloader,
            desc=f"Validation {epoch+1}/{self.epochs}",
            unit="batch",
            colour="red",
        )
        with torch.no_grad():
            for batch in loop:
                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    loss, predictions, expected = self.validation_step(batch)
                total_loss += loss.item()
                if predictions is not None:
                    all_predictions.append(predictions)
                if expected is not None:
                    all_expected.append(expected)
                loop.set_postfix(loss=f"{loss.item():.4f}")
        average_loss = total_loss / len(self.valid_dataloader)
        metrics = self.compute_metrics(all_predictions, all_expected)
        return average_loss, metrics

    def fit(self) -> None:
        best_model_state = None
        if self.monitor_mode == Extremeum.MIN:
            best_metric_value = float("inf")
            improvement = lambda current, best: current < best
        else:
            best_metric_value = float("-inf")
            improvement = lambda current, best: current > best

        for epoch in range(self.epochs):
            ColoredLogger.info(f"Starting epoch {epoch+1}/{self.epochs}")

            training_loss = self._training_epoch(epoch)
            ColoredLogger.info(f"Training loss: {training_loss:.4f}")

            validation_loss, validation_metrics = self._validation_epoch(epoch)
            ColoredLogger.info(f"Validation loss: {validation_loss:.4f}")

            for metric_name, metric_value in validation_metrics.items():
                if metric_value is None:
                    continue
                ColoredLogger.debug(f"{metric_name}: {metric_value:.4f}")   
            
            # Log epoch results
            self.history[epoch] = {
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                **{k: v for k, v in validation_metrics.items() if v is not None},
            }

            current_metric_value = self.history[epoch].get(self.monitor_metric, None)

            # Step the scheduler if provided
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_metric_value)
                else:
                    self.scheduler.step()

            # Get learning rate
            current_lr = float(
                sum(pg["lr"] for pg in self.optimizer.param_groups)
                / len(self.optimizer.param_groups)
            )

            ColoredLogger.info(f"Current learning rate: {current_lr:.6e}")
            
            # Add current learning rate to history
            self.history[epoch]["learning_rate"] = current_lr

            # Check for improvement
            if current_metric_value is not None and improvement(
                current_metric_value, best_metric_value
            ):
                best_metric_value = current_metric_value
                best_model_state = self.model.state_dict()

        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return
