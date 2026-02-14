
import random
from typing import List, TypeVar, Tuple

T = TypeVar("T")

def split_dataset(
    samples: List[T],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[T], List[T]]:
    """
    データセットを学習用と検証用に分割します。
    常に同じシードを使用することで、学習と評価で一貫した分割を保証します。
    
    Args:
        samples: 分割するデータのリスト
        train_ratio: 学習データの割合 (0.0 - 1.0)
        seed: ランダムシード
        
    Returns:
        (train_samples, val_samples)
    """
    if not samples:
        return [], []
        
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    train_size = int(len(samples) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    return train_samples, val_samples
