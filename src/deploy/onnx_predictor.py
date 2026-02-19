
import numpy as np
import onnxruntime as ort
from pathlib import Path
from src.core.types import TaskType
from dataclasses import dataclass

@dataclass
class TaskPrediction:
    label: str
    confidence: float
    logits: list[float]

@dataclass
class PredictionResult:
    cause: TaskPrediction
    shape: TaskPrediction
    depth: TaskPrediction

class ONNXPredictor:
    """ONNX Runtimeを使用した高速推論クラス"""
    
    def __init__(self, model_path: Path | str, providers: list[str] = None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
            
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # クラスラベル定義 (本来はConfig等から読み込むべき)
        # ここでは暫定的にデフォルト設定を使用
        from src.core.config import DEFAULT_CATEGORIES_CONFIG
        from src.core.category_manager import CategoryManager
        self.category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
        
    def predict(self, image_np: np.ndarray) -> PredictionResult:
        """
        推論実行
        
        Args:
            image_np: 前処理済みの画像 (H, W, C) or (C, H, W) ?
                      PyTorchのTransform出力は (C, H, W) だが、
                      DefectDataset.get_inference_transform は albumentations なので
                      (H, W, C) で返ってきて、ToTensorV2で (C, H, W) になる。
                      
                      ここへの入力は (C, H, W) の numpy array を想定 (Batch次元なし)
        """
        # Batch次元追加
        if image_np.ndim == 3:
            input_tensor = np.expand_dims(image_np, axis=0) # (1, C, H, W)
        else:
            input_tensor = image_np

        # ONNX Runtime実行
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor.astype(np.float32)}
        )
        
        # outputs は [cause_logits, shape_logits, depth_logits] のリスト
        # wrapperでの順序: cause, shape, depth
        
        result_dict = {}
        for i, task_type in enumerate([TaskType.CAUSE, TaskType.SHAPE, TaskType.DEPTH]):
            logits = outputs[i][0] # (num_classes,)
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Argmax
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            
            # Label lookup
            categories = self.category_manager.get_categories(task_type)
            if pred_idx < len(categories):
                label = categories[pred_idx]
            else:
                label = "Unknown"
                
            result_dict[task_type] = TaskPrediction(
                label=label,
                confidence=confidence,
                logits=logits.tolist()
            )
            
        return PredictionResult(
            cause=result_dict[TaskType.CAUSE],
            shape=result_dict[TaskType.SHAPE],
            depth=result_dict[TaskType.DEPTH]
        )
