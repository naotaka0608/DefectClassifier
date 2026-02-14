"""傷分類マルチタスクモデル"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .backbone import BackboneFactory
from .multi_task_head import MultiTaskHead


class DefectClassifier(nn.Module):
    """傷分類マルチタスクモデル"""

    def __init__(
        self,
        task_config: dict[str, int],
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        shared_features: int = 512,
        head_hidden_features: int = 256,
        freeze_backbone_layers: int = 6,
    ):
        super().__init__()

        self.backbone, feature_dim = BackboneFactory.create(
            name=backbone_name,
            pretrained=pretrained,
            freeze_layers=freeze_backbone_layers,
        )

        self.multi_task_head = MultiTaskHead(
            in_features=feature_dim,
            task_config=task_config,
            shared_features=shared_features,
            head_hidden_features=head_hidden_features,
            dropout=dropout,
        )

        # カテゴリ数を保存
        self.task_config = task_config

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return self.multi_task_head(features)

    def unfreeze_backbone(self, num_layers: Optional[int] = None) -> None:
        """バックボーンの凍結解除（fine-tuning用）"""
        layers = list(self.backbone.children())
        if num_layers is None:
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_config(self) -> dict:
        """モデル設定を取得"""
        return {
            "task_config": self.task_config,
            # Legacy compatibility (optional, but specific keys might be expected by old code analysis?)
            # For now, we commit to the new structure.
        }

    @classmethod
    def from_config(cls, config: dict) -> "DefectClassifier":
        """設定辞書からモデルを生成"""
        # Legacy config support
        if "task_config" not in config:
             from src.core.types import TaskType
             task_config = {
                 TaskType.CAUSE: config.get("num_cause_classes", 6),
                 TaskType.SHAPE: config.get("num_shape_classes", 3),
                 TaskType.DEPTH: config.get("num_depth_classes", 3),
             }
        else:
             task_config = config["task_config"]

        return cls(
            task_config=task_config,
            backbone_name=config.get("backbone", "resnet50"),
            pretrained=config.get("pretrained", True),
            dropout=config.get("dropout", 0.3),
            shared_features=config.get("shared_features", 512),
            head_hidden_features=config.get("head_hidden_features", 256),
            freeze_backbone_layers=config.get("freeze_backbone_layers", 6),
        )

    def save(self, path: Path | str, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """モデルを保存"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path | str, device: torch.device | str = "cpu") -> "DefectClassifier":
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=device)
        model = cls.from_config(checkpoint["config"])
        
        # State dict migration for backward compatibility
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        from src.core.types import TaskType
        
        for k, v in state_dict.items():
            # Old keys: multi_task_head.cause_head.head.0.weight
            # New keys: multi_task_head.heads.cause.head.0.weight
            if "multi_task_head.cause_head." in k:
                new_k = k.replace("multi_task_head.cause_head.", f"multi_task_head.heads.{TaskType.CAUSE}.")
            elif "multi_task_head.shape_head." in k:
                new_k = k.replace("multi_task_head.shape_head.", f"multi_task_head.heads.{TaskType.SHAPE}.")
            elif "multi_task_head.depth_head." in k:
                new_k = k.replace("multi_task_head.depth_head.", f"multi_task_head.heads.{TaskType.DEPTH}.")
            else:
                new_k = k
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict)
        return model
