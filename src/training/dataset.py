"""データセットクラス"""

import json
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
from src.core.constants import DATA_DIR, TRAIN_IMAGES_DIR
from src.core.data_manager import DataManager


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
            
        # image_pathの補完
        for sample in self.samples:
            if "image_path" not in sample and "file_name" in sample:
                rel_path = TRAIN_IMAGES_DIR.relative_to(DATA_DIR)
                sample["image_path"] = f"{rel_path}/{sample['file_name']}"


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
            "cause": self.category_manager.name_to_index("cause", sample["cause"]),
            "shape": self.category_manager.name_to_index("shape", sample["shape"]),
            "depth": self.category_manager.name_to_index("depth", sample["depth"]),
        }

        return {
            "image": image,
            "labels": {k: torch.tensor(v) for k, v in labels.items()},
            "metadata": {
                "image_path": str(image_path),
                "original_labels": {
                    "cause": sample["cause"],
                    "shape": sample["shape"],
                    "depth": sample["depth"],
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
                    A.GaussNoise(
                        var_limit=tuple(cfg.gaussian_noise["var_limit"]),
                        p=cfg.gaussian_noise["probability"],
                    ),
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
    def get_inference_transform() -> A.Compose:
        """推論用の変換を取得"""
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def collate_fn(batch: list[dict]) -> dict:
    """バッチデータをまとめる"""
    images = torch.stack([item["image"] for item in batch])
    labels = {
        "cause": torch.stack([item["labels"]["cause"] for item in batch]),
        "shape": torch.stack([item["labels"]["shape"] for item in batch]),
        "depth": torch.stack([item["labels"]["depth"] for item in batch]),
    }
    metadata = [item["metadata"] for item in batch]

    return {"image": images, "labels": labels, "metadata": metadata}
