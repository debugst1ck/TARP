from tarp.services.datasets.language.masked import MaskedLanguageModelDataset
from tarp.model.finetuning.language import LanguageModel
from tarp.services.training.trainer import Trainer

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional


class MaskedLanguageModelTrainer(Trainer):
    def __init__(
        self,
        model: LanguageModel,
        train_dataset: MaskedLanguageModelDataset,
        valid_dataset: MaskedLanguageModelDataset,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        vocabulary_size: int,
        batch_size: int = 32,
        epochs: int = 10,
        max_grad_norm: float = 1.0,
        num_workers: int = 0,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        persistent_workers: bool = False,
        criterion: Optional[nn.Module] = None,
    ):
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion
        self.criterion = self.criterion.to(device)

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
            persistent_workers=persistent_workers,
        )
        self.vocab_size = vocabulary_size

    def training_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        sequence = batch["sequence"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        truth = batch["truth"].to(self.context.device)

        # Model forward
        outputs = self.context.model(sequence, attention_mask=attention_mask)
        # Expect model to output logits of shape [batch, seq_len, vocab_size]
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]

        # Flatten for loss
        loss = self.criterion(logits.view(-1, self.vocab_size), truth.view(-1))
        return loss, logits.detach().cpu(), truth.detach().cpu()

    @torch.no_grad()
    def validation_step(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        sequence = batch["sequence"].to(self.context.device)
        attention_mask = batch["attention_mask"].to(self.context.device)
        truth = batch["truth"].to(self.context.device)

        outputs = self.context.model(sequence, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, Tensor) else outputs["logits"]

        loss = self.criterion(logits.view(-1, self.vocab_size), truth.view(-1))
        return loss, logits.detach().cpu(), truth.detach().cpu()

    def compute_metrics(
        self, prediction: list[Tensor], expected: list[Tensor], topk: int = 5
    ) -> dict[str, float]:
        logits = torch.cat(prediction)  # [total_tokens, vocab_size]
        truth = torch.cat(expected)  # [total_tokens]

        mask = truth != -100
        if mask.sum() == 0:
            return {
                "masked_accuracy": 0.0,
                f"top{topk}_accuracy": 0.0,
                "masked_perplexity": float("inf"),
            }

        predictions = logits.argmax(dim=-1)
        correct = (predictions[mask] == truth[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total

        # Top-k accuracy
        topk_predictions = torch.topk(
            logits, k=topk, dim=-1
        ).indices  # [total_tokens, topk]
        topk_correct = (
            topk_predictions[mask]
            .eq(truth[mask].unsqueeze(-1))
            .any(dim=-1)
            .sum()
            .item()
        )
        topk_accuracy = topk_correct / total

        # Perplexity
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        masked_log_probabilities = log_probabilities[range(truth.size(0)), truth][mask]
        masked_loss = -masked_log_probabilities.mean()
        masked_perplexity = torch.exp(masked_loss).item()

        return {
            "masked_accuracy": accuracy,
            f"top{topk}_accuracy": topk_accuracy,
            "masked_perplexity": masked_perplexity,
        }
