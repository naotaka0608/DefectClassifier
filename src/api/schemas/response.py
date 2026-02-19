"""レスポンススキーマ"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ClassificationResultSchema(BaseModel):
    """分類結果"""

    label: str
    confidence: float
    class_id: int
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    """推論レスポンス"""

    success: bool
    cause: ClassificationResultSchema
    shape: ClassificationResultSchema
    depth: ClassificationResultSchema
    inference_time_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    """バッチ推論レスポンス"""

    success: bool
    results: list[PredictResponse]
    total_inference_time_ms: float


class TrainStatusResponse(BaseModel):
    """学習ステータスレスポンス"""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 - 1.0
    current_epoch: int
    total_epochs: int
    metrics: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CategoriesResponse(BaseModel):
    """カテゴリ一覧レスポンス"""

    cause_categories: list[str]
    shape_categories: list[str]
    depth_categories: list[str]


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""

    status: str
    model_loaded: bool
    gpu_available: bool
    version: str


class ErrorResponse(BaseModel):
    """エラーレスポンス"""

    success: bool = False
    error: str
    detail: Optional[str] = None
