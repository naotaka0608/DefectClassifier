"""損失関数"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """マルチタスク学習用の重み付き損失関数"""

    def __init__(
        self,
        cause_weight: float = 1.0,
        shape_weight: float = 1.0,
        depth_weight: float = 1.0,
        class_weights: Optional[dict[str, torch.Tensor]] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.task_weights = {
            "cause": cause_weight,
            "shape": shape_weight,
            "depth": depth_weight,
        }
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights or {}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        損失を計算

        Returns:
            (total_loss, individual_losses): 合計損失と各タスクの損失
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for task_name in ["cause", "shape", "depth"]:
            pred = predictions[task_name]
            target = targets[task_name]

            weight = self.class_weights.get(task_name)

            task_loss = F.cross_entropy(
                pred,
                target,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )

            losses[task_name] = task_loss
            total_loss = total_loss + self.task_weights[task_name] * task_loss

        return total_loss, losses


class FocalLoss(nn.Module):
    """クラス不均衡に強いFocal Loss"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
