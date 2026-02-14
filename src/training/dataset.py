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
from src.core.constants import DATA_DIR, TRAIN_IMAGES_DIR


class DefectDataset(Dataset):
    """傷分類データセット"""

    def __init__(
        self,
        data_dir: Path | str,
        annotation_file: Path | str,
        category_manager: CategoryManager,
        transform: Optional[Callable] = None,
        is_training: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.category_manager = category_manager
        self.transform = transform or self._default_transform(is_training)

        # アノテーション読み込み
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # データ形式の互換性対応
        if isinstance(data, list):
            self.samples = data
        else:
            self.samples = data.get("samples", [])
            
        # image_pathの補完
        for sample in self.samples:
            if "image_path" not in sample and "file_name" in sample:
                # デフォルトのパス構造を仮定
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

    @staticmethod
    def _default_transform(is_training: bool) -> A.Compose:
        """デフォルトの変換を取得"""
        if is_training:
            return A.Compose(
                [
                    A.Resize(256, 256),
                    A.RandomCrop(224, 224),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    A.GaussNoise(var_limit=(10, 50), p=0.3),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(224, 224),
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
