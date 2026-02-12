"""評価メトリクス"""

from typing import Optional

import torch
import torch.nn.functional as F


class MultiTaskMetrics:
    """マルチタスク分類の評価メトリクス"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """メトリクスをリセット"""
        self._predictions: dict[str, list[torch.Tensor]] = {
            "cause": [],
            "shape": [],
            "depth": [],
        }
        self._targets: dict[str, list[torch.Tensor]] = {
            "cause": [],
            "shape": [],
            "depth": [],
        }

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        """予測と正解を追加"""
        for task_name in ["cause", "shape", "depth"]:
            pred = predictions[task_name].argmax(dim=1).cpu()
            target = targets[task_name].cpu()
            self._predictions[task_name].append(pred)
            self._targets[task_name].append(target)

    def compute(self) -> dict:
        """メトリクスを計算"""
        metrics = {}

        for task_name in ["cause", "shape", "depth"]:
            preds = torch.cat(self._predictions[task_name])
            targets = torch.cat(self._targets[task_name])

            accuracy = (preds == targets).float().mean().item()
            metrics[f"{task_name}_accuracy"] = accuracy

        # 平均精度
        metrics["mean_accuracy"] = sum(
            metrics[f"{t}_accuracy"] for t in ["cause", "shape", "depth"]
        ) / 3

        return metrics


def compute_class_weights(
    targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """クラス不均衡対応の重みを計算"""
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1)  # ゼロ除算防止
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # 正規化
    return weights


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """混同行列を計算"""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, target in zip(predictions, targets):
        matrix[target, pred] += 1
    return matrix


def compute_precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    """クラスごとの精度・再現率・F1スコアを計算"""
    confusion = compute_confusion_matrix(predictions, targets, num_classes)

    # True Positives
    tp = confusion.diag().float()
    # False Positives
    fp = confusion.sum(dim=0).float() - tp
    # False Negatives
    fn = confusion.sum(dim=1).float() - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion,
    }
