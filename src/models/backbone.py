"""バックボーンネットワーク"""

from typing import Literal

import torch.nn as nn
from torchvision import models


class BackboneFactory:
    """バックボーンネットワーク生成ファクトリ"""

    @staticmethod
    def create(
        name: Literal["resnet50", "resnet101", "efficientnet_b4"] = "resnet50",
        pretrained: bool = True,
        freeze_layers: int = 6,
    ) -> tuple[nn.Module, int]:
        """
        バックボーンを生成

        Args:
            name: バックボーン名
            pretrained: 事前学習済み重みを使用するか
            freeze_layers: 凍結する層数

        Returns:
            (backbone, feature_dim): バックボーンモジュールと出力特徴次元
        """
        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {name}")

        if freeze_layers > 0:
            BackboneFactory._freeze_layers(backbone, freeze_layers)

        return backbone, feature_dim

    @staticmethod
    def _freeze_layers(model: nn.Module, num_layers: int) -> None:
        """指定した数の層を凍結"""
        layers = list(model.children())
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = False
