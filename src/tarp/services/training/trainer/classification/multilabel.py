import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional

from tarp.services.training.trainer import Trainer
from tarp.services.evaluation import Extremum
from tarp.services.evaluation.classification.multilabel import MultiLabelMetrics
from tarp.services.evaluation.losses.multilabel import AsymmetricFocalLoss

from tarp.model.finetuning.classification import ClassificationModel

class MultiLabelClassificationTrainer(Trainer):
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
        class_weights: Optional[Tensor] = None,
        criterion: Optional[nn.Module] = None,
        accumulation_steps: int = 1,
    ):
        if criterion is None:
            if class_weights is not None:
                # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
                self.criterion = AsymmetricFocalLoss(
                    gamma_neg=2, gamma_pos=0, class_weights=class_weights.to(device)
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

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
        )
        
        self.criterion = self.criterion.to(device)
        
        self.metrics = MultiLabelMetrics()
        
    def training_step(self, batch: dict[str, Tensor]) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()
    
    def validation_step(self, batch: dict[str, Tensor]) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        inputs = batch["sequence"].to(self.context.device)
        labels = batch["labels"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        logits: Tensor = self.context.model(inputs, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()
    
    def compute_metrics(self, logits, labels):
        return self.metrics.compute(logits, labels)