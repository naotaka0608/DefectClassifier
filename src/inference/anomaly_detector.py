"""異常検知・未知データ検知モジュール"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class AnomalyDetector:
    """入力データが『未知』のものかどうかを判定するクラス"""

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def detect(self, logits_dict: Dict[str, torch.Tensor]) -> Tuple[bool, float]:
        """
        全タスクのロジットから確信度を評価し、異常（未知）かどうかを判定する。
        
        Args:
            logits_dict: タスク名をキー、ロジットテンソルを値とする辞書
            
        Returns:
            is_anomaly: 異常な場合に True
            score: 確信度の最小値（低いほど異常の可能性が高い）
        """
        # 単純な最大確率（Softmax確信度）を用いた手法
        confidences = []
        for task_name, logits in logits_dict.items():
            probs = F.softmax(logits, dim=1)
            max_prob = torch.max(probs).item()
            confidences.append(max_prob)
        
        # 最も確信度が低いタスクの値をスコアとする
        min_conf = min(confidences) if confidences else 1.0
        is_anomaly = min_conf < self.threshold
        
        return is_anomaly, min_conf

    def detect_by_features(self, features: torch.Tensor) -> float:
        """
        (拡張用) バックボーンの特徴量ベクトル（GAP後）を用いた異常検知
        """
        # 現状はプレースホルダ。将来的に孤立森(Isolation Forest)などを利用可能
        return 0.0
