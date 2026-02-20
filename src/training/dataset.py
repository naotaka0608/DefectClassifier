"""データセットクラス"""

from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.core.category_manager import CategoryManager
from src.core.config import AugmentationConfig
from src.core.data_manager import DataManager
from src.core.types import TaskType


class DefectDataset(Dataset):
    """傷分類データセット"""

    def __init__(
        self,
        data_dir: Path | str,
        category_manager: CategoryManager,
        annotation_file: Optional[Path | str] = None,
        samples: Optional[list[dict]] = None,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        aug_config: Optional[AugmentationConfig] = None,
    ):
        self.data_dir = Path(data_dir)
        self.category_manager = category_manager
        self.aug_config = aug_config or AugmentationConfig()
        self.transform = transform or self._default_transform(is_training)

        # サンプルの取得: 直接渡されたリストを優先、なければファイルから読み込み
        if samples is not None:
            self.samples = samples
        elif annotation_file is not None:
            data_manager = DataManager(annotation_file)
            self.samples = data_manager.load_annotations()
        else:
            raise ValueError("annotation_file か samples のどちらかを指定してください")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 画像読み込み
        image_path = self.data_dir / sample["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # 変換適用
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # ラベルをインデックスに変換
        labels = {
            TaskType.CAUSE: self.category_manager.name_to_index(TaskType.CAUSE, sample["cause"]),
            TaskType.SHAPE: self.category_manager.name_to_index(TaskType.SHAPE, sample["shape"]),
            TaskType.DEPTH: self.category_manager.name_to_index(TaskType.DEPTH, sample["depth"]),
        }

        return {
            "image": image,
            "labels": {k: torch.tensor(v) for k, v in labels.items()},
            "metadata": {
                "image_path": str(image_path),
                "original_labels": {
                    TaskType.CAUSE: sample["cause"],
                    TaskType.SHAPE: sample["shape"],
                    TaskType.DEPTH: sample["depth"],
                },
            },
        }

    def _default_transform(self, is_training: bool) -> A.Compose:
        """AugmentationConfig に基づいた変換を構築"""
        cfg = self.aug_config
        if is_training:
            return A.Compose(
                [
                    A.Resize(cfg.resize[0], cfg.resize[1]),
                    A.RandomCrop(cfg.crop_size[0], cfg.crop_size[1]),
                    A.HorizontalFlip(p=cfg.horizontal_flip),
                    A.VerticalFlip(p=cfg.vertical_flip),
                    A.RandomRotate90(p=cfg.random_rotate90),
                    A.ColorJitter(
                        brightness=cfg.color_jitter["brightness"],
                        contrast=cfg.color_jitter["contrast"],
                        saturation=cfg.color_jitter["saturation"],
                        hue=cfg.color_jitter["hue"],
                    ),
                    # A.GaussNoise(
                    #     var_limit=tuple(cfg.gaussian_noise["var_limit"]),
                    #     p=cfg.gaussian_noise["probability"],
                    # ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(cfg.crop_size[0], cfg.crop_size[1]),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ]
            )


    @staticmethod
    def get_inference_transform(config: Optional[AugmentationConfig] = None, image_size: Optional[list[int]] = None) -> A.Compose:
        """推論用の変換を取得
        
        Args:
            config: AugmentationConfig (optional)
            image_size: [height, width] (optional, overrides config)
        """
        # デフォルトサイズ
        height, width = 224, 224
        
        if image_size is not None:
            height, width = image_size[0], image_size[1]
        elif config is not None:
            # AugmentationConfig usually has crop_size or resize. 
            # For inference, we typically want the target input size.
            # Using crop_size as default target size if available.
            height, width = config.crop_size[0], config.crop_size[1]

        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def collate_fn(batch: list[dict]) -> dict:
    """バッチデータをまとめる"""
    images = torch.stack([item["image"] for item in batch])
    labels = {
        TaskType.CAUSE: torch.stack([item["labels"][TaskType.CAUSE] for item in batch]),
        TaskType.SHAPE: torch.stack([item["labels"][TaskType.SHAPE] for item in batch]),
        TaskType.DEPTH: torch.stack([item["labels"][TaskType.DEPTH] for item in batch]),
    }
    metadata = [item["metadata"] for item in batch]

    return {"image": images, "labels": labels, "metadata": metadata}
