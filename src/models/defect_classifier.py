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
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        num_cause_classes: int = 6,
        num_shape_classes: int = 3,
        num_depth_classes: int = 3,
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
            shared_features=shared_features,
            head_hidden_features=head_hidden_features,
            num_cause_classes=num_cause_classes,
            num_shape_classes=num_shape_classes,
            num_depth_classes=num_depth_classes,
            dropout=dropout,
        )

        # カテゴリ数を保存
        self.num_cause_classes = num_cause_classes
        self.num_shape_classes = num_shape_classes
        self.num_depth_classes = num_depth_classes

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
            "num_cause_classes": self.num_cause_classes,
            "num_shape_classes": self.num_shape_classes,
            "num_depth_classes": self.num_depth_classes,
        }

    @classmethod
    def from_config(cls, config: dict) -> "DefectClassifier":
        """設定辞書からモデルを生成"""
        return cls(
            backbone_name=config.get("backbone", "resnet50"),
            pretrained=config.get("pretrained", True),
            num_cause_classes=config["num_cause_classes"],
            num_shape_classes=config["num_shape_classes"],
            num_depth_classes=config["num_depth_classes"],
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
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
