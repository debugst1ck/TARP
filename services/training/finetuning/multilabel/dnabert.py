from services.loggers.colored import ColoredLogger
from services.tokenizers.pretrained.dnabert import Dnabert2Tokenizer
from model.finetuning.classification.dnabert2 import SimpleClassifier
from services.evalutation.classification.multilabel import MultilabelMetrics

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, StepLR

from tqdm import tqdm

from typing import Optional
from pathlib import Path


class Dnabert2FinetuningClassificationTrainer:
    def __init__(
        self,
        model: SimpleClassifier,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
    ):
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

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

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        loop = tqdm(
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
        results = metrics.compute(all_labels, all_logits)
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

            for metric_name, metric_value in metrics.items():
                if metric_value is None:
                    continue
                ColoredLogger.debug(f"{metric_name.capitalize()}: {metric_value:.4f}")

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
