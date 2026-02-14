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


from src.core.types import TaskType

class MultiTaskHead(nn.Module):
    """マルチタスク分類ヘッド"""

    def __init__(
        self,
        in_features: int,
        task_config: dict[str, int],
        shared_features: int = 512,
        head_hidden_features: int = 256,
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
        self.heads = nn.ModuleDict()
        for task_name, num_classes in task_config.items():
            self.heads[task_name] = TaskHead(
                shared_features, head_hidden_features, num_classes, dropout
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_layer(x)
        
        return {
            task_name: head(shared)
            for task_name, head in self.heads.items()
        }
