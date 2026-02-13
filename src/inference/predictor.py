"""推論クラス"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.core.category_manager import CategoryManager
from src.models.defect_classifier import DefectClassifier


@dataclass
class ClassificationResult:
    """分類結果"""

    label: str
    confidence: float
    class_id: int


@dataclass
class PredictionResult:
    """推論結果"""

    cause: ClassificationResult
    shape: ClassificationResult
    depth: ClassificationResult


class DefectPredictor:
    """傷分類推論クラス"""

    def __init__(
        self,
        model_path: Path | str,
        category_manager: CategoryManager,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.category_manager = category_manager

        # モデル読み込み
        self.model = DefectClassifier.load(model_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        # 前処理
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )

        self.model_version = Path(model_path).stem

    def reload_model(self, model_path: Path | str) -> None:
        """モデルを再読み込み"""
        if Path(model_path).stem == self.model_version:
             return

        print(f"Reloading model from {model_path}...")
        self.model = DefectClassifier.load(model_path, self.device)
        self.model.to(self.device)
        self.model.eval()
        self.model_version = Path(model_path).stem
        print(f"Model reloaded: {self.model_version}")

    def _decode_image(self, image_base64: str) -> np.ndarray:
        """Base64画像をデコード"""
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """前処理"""
        transformed = self.transform(image=image)
        return transformed["image"].unsqueeze(0).to(self.device)

    def _postprocess(
        self, outputs: dict[str, torch.Tensor], return_confidence: bool = True
    ) -> PredictionResult:
        """後処理"""
        results = {}

        for task_name, logits in outputs.items():
            probs = F.softmax(logits, dim=1)
            confidence, class_id = probs.max(dim=1)

            class_id = class_id.item()
            confidence = confidence.item()

            label = self.category_manager.index_to_name(task_name, class_id)

            results[task_name] = ClassificationResult(
                label=label,
                confidence=confidence if return_confidence else 0.0,
                class_id=class_id,
            )

        return PredictionResult(
            cause=results["cause"],
            shape=results["shape"],
            depth=results["depth"],
        )

    @torch.no_grad()
    def predict_from_base64(
        self,
        image_base64: str,
        return_confidence: bool = True,
    ) -> PredictionResult:
        """Base64画像から推論"""
        image = self._decode_image(image_base64)
        return self.predict(image, return_confidence)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_confidence: bool = True,
    ) -> PredictionResult:
        """画像から推論"""
        input_tensor = self._preprocess(image)
        outputs = self.model(input_tensor)
        return self._postprocess(outputs, return_confidence)

    @torch.no_grad()
    def predict_from_file(
        self,
        image_path: Path | str,
        return_confidence: bool = True,
    ) -> PredictionResult:
        """ファイルから推論"""
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        return self.predict(image, return_confidence)

    @torch.no_grad()
    def predict_batch(
        self,
        images: list[np.ndarray],
        return_confidence: bool = True,
    ) -> list[PredictionResult]:
        """バッチ推論"""
        batch_tensors = []
        for image in images:
            tensor = self._preprocess(image)
            batch_tensors.append(tensor)

        batch_input = torch.cat(batch_tensors, dim=0)
        outputs = self.model(batch_input)

        results = []
        batch_size = len(images)

        for i in range(batch_size):
            single_outputs = {
                task: logits[i : i + 1] for task, logits in outputs.items()
            }
            result = self._postprocess(single_outputs, return_confidence)
            results.append(result)

        return results

    def get_all_probabilities(
        self, image: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """全クラスの確率を取得"""
        input_tensor = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        result = {}
        for task_name, logits in outputs.items():
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            categories = self.category_manager.get_categories(task_name)
            result[task_name] = {
                cat: float(prob) for cat, prob in zip(categories, probs)
            }

        return result
