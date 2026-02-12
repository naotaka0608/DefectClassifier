"""マルチタスク分類ヘッド"""

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """単一タスク用の分類ヘッド"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskHead(nn.Module):
    """マルチタスク分類ヘッド"""

    def __init__(
        self,
        in_features: int,
        shared_features: int = 512,
        head_hidden_features: int = 256,
        num_cause_classes: int = 6,
        num_shape_classes: int = 3,
        num_depth_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 共有特徴層
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features, shared_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # タスク別ヘッド
        self.cause_head = TaskHead(
            shared_features, head_hidden_features, num_cause_classes, dropout
        )
        self.shape_head = TaskHead(
            shared_features, head_hidden_features, num_shape_classes, dropout
        )
        self.depth_head = TaskHead(
            shared_features, head_hidden_features, num_depth_classes, dropout
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_layer(x)

        return {
            "cause": self.cause_head(shared),
            "shape": self.shape_head(shared),
            "depth": self.depth_head(shared),
        }
