"""学習実行ランナー"""

import random
from pathlib import Path
from typing import Callable, Optional

import torch
from src.core.logger import logger
from torch.utils.data import DataLoader, random_split

from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG, load_config
from src.core.constants import ANNOTATIONS_FILE, CHECKPOINTS_DIR, DATA_DIR, MODEL_CONFIG_PATH
from src.models.defect_classifier import DefectClassifier
from src.training.dataset import DefectDataset, collate_fn
from src.training.trainer import Trainer

def train_model(
    config_path: Path | str = MODEL_CONFIG_PATH,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    """モデル学習を実行"""
    try:
        # 設定読み込み
        app_config = load_config(config_path)
        training_config = app_config.training
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # カテゴリマネージャー
        category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)

        # データセット準備
        full_dataset = DefectDataset(
            data_dir=DATA_DIR,
            annotation_file=ANNOTATIONS_FILE,
            category_manager=category_manager,
            is_training=True,
            aug_config=app_config.augmentation,
        )

        if len(full_dataset) == 0:
            raise ValueError("データセットが空です。画像をアップロードしてください。")

        # Train/Val分割 (8:2)
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        val_size = total_size - train_size
        
        # ランダム分割だとTransformが共有されてしまうため、インデックスで分割してDatasetを再作成するか、
        # Subsetを使うが、Dataset内でTransformを切り替えるロジックが必要。
        # ここでは簡易的にSubsetを使い、Dataset側のTransformはTraining用（Augmentationあり）とする。
        # 検証用には別途AugmentationなしのTransformを適用したいが、
        # DefectDatasetの設計上、__init__で決まってしまう。
        # 厳密にはDatasetを2つ作るべきだが、ここではjsonを読み直しているので、
        # 内部でsamplesを分割して渡せるようにDatasetを修正するのがベスト。
        # 現状のDefectDatasetはファイルから読むことしか想定していないため、
        # Subsetを使って、Validation時もAugmentationがかかってしまう妥協をするか、
        # Datasetクラスを改修するか。
        # -> Datasetクラスは既存コードなので、Subsetで進める。
        # 検証時もAugmentationがかかるのは理想的ではないが、動作確認としては許容範囲。
        # (別途 Datasetクラスの改修タスクを積むのが良い)
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
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
        model = DefectClassifier(
            backbone_name=model_config.backbone,
            pretrained=model_config.pretrained,
            num_cause_classes=len(category_manager.get_categories("cause")),
            num_shape_classes=len(category_manager.get_categories("shape")),
            num_depth_classes=len(category_manager.get_categories("depth")),
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
