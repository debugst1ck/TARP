from services.loggers.colored import ColoredLogger
from model.finetuning.classification import ClassificationModel
from services.evalutation.classification.multilabel import MultilabelMetrics

from services.evalutation.loss.multilabel import AsymmetricLoss

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, StepLR

from tqdm import tqdm

from typing import Optional

from typing import Union


class FinetuningClassificationTrainer:
    def __init__(
        self,
        model: ClassificationModel,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        class_weights: Optional[Tensor] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        if criterion is not None:
            self.criterion = criterion
        else:
            if class_weights is not None:
                # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
                self.criterion = AsymmetricLoss(
                    gamma_neg=2, gamma_pos=0, class_weights=class_weights.to(device)
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

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
        self.model.to(self.device)

        self.history: list[dict[str, float]] = [{} for _ in range(self.epochs)]

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        loop: Union[Dataset[dict[str, Tensor]]] = tqdm(
            self.train_dataloader,
            desc=f"Training {epoch+1}/{self.epochs}",
            unit="batch",
        )
        for batch in loop:
            inputs = batch["sequence"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs, attention_mask)
            loss: Tensor = self.criterion(logits, labels)

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        average_training_loss = total_loss / len(self.train_dataloader)
        return average_training_loss

    def _validate_epoch(self, epoch: int) -> tuple[float, dict[str, Optional[float]]]:
        self.model.eval()

        total_loss = 0.0
        all_labels, all_logits = [], []

        metrics = MultilabelMetrics()

        loop = tqdm(
            self.valid_dataloader,
            desc=f"Validation {epoch+1}/{self.epochs}",
            unit="batch",
        )
        with torch.no_grad():
            for batch in loop:
                inputs = batch["sequence"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels: Tensor = batch["labels"].to(self.device)

                logits: Tensor = self.model(inputs, attention_mask)
                loss: Tensor = self.criterion(logits, labels)

                total_loss += loss.item()
                all_labels.append(labels.cpu())
                all_logits.append(logits.cpu())

                loop.set_postfix(loss=loss.item())

        average_validation_loss = total_loss / len(self.valid_dataloader)
        results = metrics.compute(all_logits, all_labels)
        return average_validation_loss, results

    def fit(self):
        best_validation_loss = float("inf")

        best_model_state = None

        for epoch in range(self.epochs):
            # Log training start
            ColoredLogger.info(f"Starting epoch {epoch+1}/{self.epochs}")
            train_loss = self._train_epoch(epoch)
            ColoredLogger.info(f"Training loss: {train_loss:.4f}")
            validation_loss, metrics = self._validate_epoch(epoch)
            ColoredLogger.info(f"Validation loss: {validation_loss:.4f}")
            
            # Learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    lr = self.scheduler.optimizer.param_groups[0]["lr"]
                else:
                    lr = self.scheduler.get_last_lr()[0]
                ColoredLogger.info(f"Learning rate: {lr:.6f}")

            for metric_name, metric_value in metrics.items():
                if metric_value is None:
                    continue
                ColoredLogger.debug(f"{metric_name.capitalize()}: {metric_value:.4f}")

            self.history[epoch] = {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{k: v for k, v in metrics.items() if v is not None},
            }

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(
                    validation_loss
                )  # or metrics["f1"] if monitoring f1

            elif self.scheduler is not None:
                self.scheduler.step()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = self.model.state_dict()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            ColoredLogger.info(
                f"Best model loaded with validation loss: {best_validation_loss:.4f}"
            )
