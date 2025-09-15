from torch import nn, Tensor
import torch

from typing import Optional


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
        reduction: str = "mean",  # add reduction to mimic BCE
        class_weights: Optional[Tensor] = None,
    ):
        """
        Asymmetric Loss for multi-label classification.

        :param float gamma_neg: focusing parameter for negative samples, higher values put more focus on hard negatives
        :param float gamma_pos: focusing parameter for positive samples, higher values put more focus on hard positives
        :param float clip: if > 0, adds a small value to the negative logits before applying sigmoid, helps with extreme negatives
        :param float eps: small value to avoid log(0)
        :param bool disable_torch_grad_focal_loss: if True, disables gradient computation for focal loss part to save memory
        :param str reduction: reduction method to apply to the output loss ('none', 'mean', 'sum')
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric Loss for multi-label classification.
        Args:
            logits: raw model outputs (before sigmoid), shape (batch, num_labels)
            targets: binary targets, same shape
        """
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg: Tensor = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg: Tensor = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # Apply class weights
        if self.class_weights is not None:
            loss *= self.class_weights.unsqueeze(0)  # Broadcast to match batch size

        # Apply reduction
        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
