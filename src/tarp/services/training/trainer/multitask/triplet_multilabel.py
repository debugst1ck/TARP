import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional

from tarp.cli.logging import Console
from tarp.model.finetuning.classification import ClassificationModel
from tarp.services.training.callbacks import Callback
from tarp.services.training.callbacks.monitoring import (
    EarlyStopping,
    LearningRateScheduler,
)
from tarp.services.training.trainer import Trainer
from tarp.services.evaluation.classification.multilabel import MultiLabelMetrics
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss
from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import classification_report

import torch.nn.functional as F


class JointTripletClassificationTrainer(Trainer):
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
        use_amp: bool = True,
        accumulation_steps: int = 1,
        callbacks: list[Callback] = [
            EarlyStopping(5),
            LearningRateScheduler("validation_loss"),
        ],
        class_weights: Optional[Tensor] = None,
        shared: dict = {},
        lambda_classification: float = 1.0,
        lambda_triplet: float = 0.1,
    ) -> None:
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            use_amp=use_amp,
            accumulation_steps=accumulation_steps,
            callbacks=callbacks,
            shared=shared,
        )

        # Emphasize on rare positives and ignore easy negatives
        self.classification_loss = AsymmetricFocalLoss(gamma_pos=2.0, gamma_neg=2.0, class_weights=class_weights)

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

        self.lambda_classification = lambda_classification
        self.lambda_triplet = lambda_triplet
        self.metrics = MultiLabelMetrics()

    def _move_to_device(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            k: v.to(self.context.device)
            for k, v in x.items()
            if isinstance(v, torch.Tensor)
        }

    def training_step(
        self, batch: dict[str, dict[str, Tensor]]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # Move to device
        anchor = self._move_to_device(batch["anchor"])
        positive = self._move_to_device(batch["positive"])
        negative = self._move_to_device(batch["negative"])

        model: ClassificationModel = self.context.model

        # Forward pass through encoder
        anchor_embeddings = model.encoder.encode(
            anchor["sequence"], anchor.get("attention_mask")
        )
        positive_embeddings = model.encoder.encode(
            positive["sequence"], positive.get("attention_mask")
        )
        negative_embeddings = model.encoder.encode(
            negative["sequence"], negative.get("attention_mask")
        )

        # Classification logits
        anchor_logits: Tensor = model.classification_head(anchor_embeddings)
        positive_logits: Tensor = model.classification_head(positive_embeddings)
        negative_logits: Tensor = model.classification_head(negative_embeddings)

        # Classification loss (multi-label)
        anchor_labels = anchor["labels"]
        positive_labels = positive["labels"]
        negative_labels = negative["labels"]

        classification_loss = (
            self.classification_loss(anchor_logits, anchor_labels)
            + self.classification_loss(positive_logits, positive_labels)
            + self.classification_loss(negative_logits, negative_labels)
        ) / 3.0

        # Triplet loss
        triplet_loss = self.triplet_loss(
            F.normalize(anchor_embeddings),
            F.normalize(positive_embeddings),
            F.normalize(negative_embeddings),
        )

        total_loss = (
            self.lambda_classification * classification_loss
            + self.lambda_triplet * triplet_loss
        )

        return total_loss, anchor_logits.detach().cpu(), anchor_labels.detach().cpu()

    def validation_step(
        self, batch: dict[str, dict[str, Tensor]]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        anchor = self._move_to_device(batch["anchor"])
        model: ClassificationModel = self.context.model
        anchor_emb = model.encoder.encode(
            anchor["sequence"], anchor.get("attention_mask")
        )
        logits: Tensor = model.classification_head(anchor_emb)
        loss = self.classification_loss(logits, anchor["labels"])
        return loss, logits.detach().cpu(), anchor["labels"].detach().cpu()

    def compute_metrics(self, prediction, expected):
        Console.debug(
            "Validation Classification Report:\n"
            + classification_report(
                torch.concat(expected, dim=0).numpy(),
                (torch.sigmoid(torch.concat(prediction, dim=0)) > 0.5).numpy(),
                zero_division=0,
            )
        )
        return self.metrics.compute(prediction, expected)
