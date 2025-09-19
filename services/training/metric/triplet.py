from model.finetuning.metric.triplet import TripletMetricModel
from services.loggers.colored import ColoredLogger

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler

from tqdm import tqdm
from typing import Optional

from services.training import Trainer

class TripletMetricTrainer(Trainer):
    def __init__(
        self,
        model: TripletMetricModel,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        margin: float = 1.0,
        num_workers: int = 0,
        criterion: Optional[nn.Module] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        # Loss function
        self.criterion = (
            criterion if criterion is not None else nn.TripletMarginLoss(margin=margin)
        )

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

    def _move_item_to_device(
        self, item: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor]]:
        sequence = item["sequence"].to(self.device)
        mask = item.get("mask", None)
        if mask is not None:
            mask = mask.to(self.device)
        return sequence, mask

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        loop = tqdm(
            self.train_dataloader,
            desc=f"Training {epoch+1}/{self.epochs}",
            unit="batch",
        )

        for batch in loop:
            # Move data to device
            anchor, anchor_mask = self._move_item_to_device(batch["anchor"])
            positive, positive_mask = self._move_item_to_device(batch["positive"])
            negative, negative_mask = self._move_item_to_device(batch["negative"])

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            anchor_embeddings, positive_embeddings, negative_embeddings = self.model(
                anchor,
                positive,
                negative,
                anchor_mask=anchor_mask,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
            )
            loss: Tensor = self.criterion(
                anchor_embeddings, positive_embeddings, negative_embeddings
            )

            loss.backward()

            # Gradient clipping
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        average_loss = total_loss / len(self.train_dataloader)
        return average_loss

    def _validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        loop = tqdm(
            self.valid_dataloader,
            desc=f"Validation {epoch+1}/{self.epochs}",
            unit="batch",
        )

        with torch.no_grad():
            for batch in loop:
                # Move data to device
                anchor, anchor_mask = self._move_item_to_device(batch["anchor"])
                positive, positive_mask = self._move_item_to_device(batch["positive"])
                negative, negative_mask = self._move_item_to_device(batch["negative"])

                # Forward pass
                anchor_embeddings, positive_embeddings, negative_embeddings = self.model(
                    anchor,
                    positive,
                    negative,
                    anchor_mask=anchor_mask,
                    positive_mask=positive_mask,
                    negative_mask=negative_mask,
                )
                
                loss: Tensor = self.criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        average_loss = total_loss / len(self.valid_dataloader)
        return average_loss
    
    def fit(self):
        best_validation_loss = float("inf")
        best_model_state = None

        for epoch in range(self.epochs):
            ColoredLogger.info(f"Starting epoch {epoch+1}/{self.epochs}")
            train_loss = self._train_epoch(epoch)
            ColoredLogger.info(f"Training loss: {train_loss:.4f}")

            validation_loss = self._validate_epoch(epoch)
            ColoredLogger.info(f"Validation loss: {validation_loss:.4f}")

            # Learning rate logging
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    lr = self.scheduler.optimizer.param_groups[0]["lr"]
                else:
                    lr = self.scheduler.get_last_lr()[0]
                ColoredLogger.info(f"Learning rate: {lr:.6f}")

            self.history[epoch] = {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(validation_loss)
                else:
                    self.scheduler.step()

            # Save best
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = self.model.state_dict()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            ColoredLogger.info(f"Best model loaded with validation loss: {best_validation_loss:.4f}")
