"""学習実行ランナー"""

import random
from pathlib import Path
from typing import Callable, Optional

import torch
from src.core.logger import logger
from torch.utils.data import DataLoader

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG, load_config
from src.core.constants import ANNOTATIONS_FILE, CHECKPOINTS_DIR, DATA_DIR, MODEL_CONFIG_PATH
from src.core.data_manager import DataManager
from src.models.defect_classifier import DefectClassifier
from src.training.dataset import DefectDataset, collate_fn
from src.training.trainer import Trainer
from src.core.types import TaskType

def train_model(
    config_path: Path | str = MODEL_CONFIG_PATH,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    """モデル学習を実行"""
    try:
        # 設定読み込み
        app_config = load_config(config_path)
        training_config = app_config.training
        aug_config = app_config.augmentation
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # カテゴリマネージャー
        category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

        # サンプルを一度だけ読み込み
        data_manager = DataManager(ANNOTATIONS_FILE)
        all_samples = data_manager.load_annotations()

        if len(all_samples) == 0:
            raise ValueError("データセットが空です。画像をアップロードしてください。")

        # インデックスベースで Train/Val 分割 (8:2)
        # 共通の分割ロジックを使用
        from src.core.data_utils import split_dataset
        train_samples, val_samples = split_dataset(all_samples, train_ratio=0.8, seed=42)

        # Train: Augmentation あり / Val: Augmentation なし
        train_dataset = DefectDataset(
            data_dir=DATA_DIR,
            category_manager=category_manager,
            samples=train_samples,
            is_training=True,
            aug_config=aug_config,
        )
        val_dataset = DefectDataset(
            data_dir=DATA_DIR,
            category_manager=category_manager,
            samples=val_samples,
            is_training=False,
            aug_config=aug_config,
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0, # Windowsでのmultiprocessingエラー回避のため0推奨(または if __name__ == '__main__': ガードが必要)
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if device.type == "cuda" else False
        )

        # モデル初期化
        model_config = app_config.model
        
        task_config = {
            TaskType.CAUSE: len(category_manager.get_categories(TaskType.CAUSE)),
            TaskType.SHAPE: len(category_manager.get_categories(TaskType.SHAPE)),
            TaskType.DEPTH: len(category_manager.get_categories(TaskType.DEPTH)),
        }
        
        model = DefectClassifier(
            task_config=task_config,
            backbone_name=model_config.backbone,
            pretrained=model_config.pretrained,
            dropout=model_config.dropout,
            shared_features=model_config.shared_features,
            head_hidden_features=model_config.head_hidden_features,
            freeze_backbone_layers=model_config.freeze_backbone_layers,
        )

        # トレーナー初期化
        trainer = Trainer(
            model=model,
            config=training_config,
            device=device,
            checkpoint_dir=CHECKPOINTS_DIR
        )

        # 学習実行
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            progress_callback=progress_callback
        )
        
        logger.info("Training completed.")
        return history

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
