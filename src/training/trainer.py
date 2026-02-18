"""学習トレーナー"""

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from src.core.logger import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import TrainingConfig
from src.core.types import TaskType
from src.models.defect_classifier import DefectClassifier
from src.models.losses import MultiTaskLoss

from .metrics import MultiTaskMetrics


class EarlyStopping:
    """早期停止コールバック"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """マルチタスク学習トレーナー"""

    def __init__(
        self,
        model: DefectClassifier,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path | str,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 最適化設定
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.criterion = MultiTaskLoss(
            task_weights=config.loss_weights,
            label_smoothing=config.label_smoothing,
        )

        self.scaler = (
            torch.amp.GradScaler("cuda") if config.mixed_precision and device.type == "cuda" else None
        )

        self.early_stopping = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )
        
        # クラス数をモデルから取得してメトリクスに渡す
        # model.task_config は {task_name: num_classes(int)} の辞書
        num_classes = model.task_config.copy()
        self.metrics = MultiTaskMetrics(num_classes=num_classes)

        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """学習実行"""
        history = {"train_loss": [], "val_loss": [], "metrics": []}

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # 学習フェーズ
            train_loss = self._train_epoch(train_loader)

            # 検証フェーズ
            val_loss, val_metrics = self._validate_epoch(val_loader)

            # 記録
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["metrics"].append(val_metrics)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Mean Acc: {val_metrics['mean_accuracy']:.4f}"
            )

            # コールバック
            if progress_callback:
                progress_callback(
                    {
                        "epoch": epoch + 1,
                        "total_epochs": self.config.epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "metrics": val_metrics,
                    }
                )

            # スケジューラ更新
            self.scheduler.step()

            # ベストモデル保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pth")

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # 早期停止チェック
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # 最終モデル保存
        self._save_checkpoint("final_model.pth")

        return history

    def _train_step(self, batch: dict) -> float:
        """1バッチの学習ステップ"""
        images = batch["image"].to(self.device)
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

        self.optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=self.scaler is not None):
            predictions = self.model(images)
            loss, _ = self.criterion(predictions, labels)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

        return loss.item()

    def _validate_step(self, batch: dict) -> tuple[float, dict, dict]:
        """1バッチの検証ステップ"""
        images = batch["image"].to(self.device)
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

        with torch.amp.autocast("cuda", enabled=self.scaler is not None):
            predictions = self.model(images)
            loss, _ = self.criterion(predictions, labels)

        return loss.item(), predictions, labels

    def _train_epoch(self, loader: DataLoader) -> float:
        """1エポック学習"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Training Epoch {self.current_epoch + 1}")
        for batch in pbar:
            loss = self._train_step(batch)
            total_loss += loss
            pbar.set_postfix({"loss": loss})

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        self.metrics.reset()

        for batch in tqdm(loader, desc="Validation"):
            loss, predictions, labels = self._validate_step(batch)
            total_loss += loss
            self.metrics.update(predictions, labels)

        return total_loss / len(loader), self.metrics.compute()

    def _save_checkpoint(self, filename: str) -> None:
        """チェックポイント保存"""
        path = self.checkpoint_dir / filename
        self.model.save(path, self.optimizer)
        logger.info(f"Saved checkpoint: {path}")
