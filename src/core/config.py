"""設定読み込みモジュール"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    """モデル設定"""

    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone_layers: int = 6
    dropout: float = 0.3
    shared_features: int = 512
    head_hidden_features: int = 256


class TrainingConfig(BaseModel):
    """学習設定"""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    warmup_epochs: int = 5
    patience: int = 10
    min_delta: float = 0.001
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    loss_weights: dict[str, float] = {"cause": 1.0, "shape": 1.0, "depth": 1.0}
    label_smoothing: float = 0.1


class AugmentationConfig(BaseModel):
    """データ拡張設定"""

    resize: list[int] = [256, 256]
    crop_size: list[int] = [224, 224]
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    random_rotate90: float = 0.5
    color_jitter: dict[str, float] = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
    }
    gaussian_noise: dict[str, Any] = {"var_limit": [10, 50], "probability": 0.3}


class InferenceConfig(BaseModel):
    """推論設定"""

    image_size: list[int] = [224, 224]
    batch_size: int = 16
    default_threshold: float = 0.5


class AppConfig(BaseModel):
    """アプリケーション全体設定"""

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    inference: InferenceConfig = InferenceConfig()


def load_config(config_path: Path | str) -> AppConfig:
    """設定ファイルを読み込み"""
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return AppConfig(
        model=ModelConfig(**raw_config.get("model", {})),
        training=TrainingConfig(**raw_config.get("training", {})),
        augmentation=AugmentationConfig(**raw_config.get("augmentation", {})),
        inference=InferenceConfig(**raw_config.get("inference", {})),
    )


# デフォルト設定パス
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DEFAULT_MODEL_CONFIG = DEFAULT_CONFIG_DIR / "model_config.yaml"
DEFAULT_CATEGORIES_CONFIG = DEFAULT_CONFIG_DIR / "categories.yaml"
