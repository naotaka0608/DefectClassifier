"""リクエストスキーマ"""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """単一画像推論リクエスト"""

    image_base64: str = Field(..., description="Base64エンコードされた画像")
    return_confidence: bool = Field(default=True, description="信頼度スコアを返すか")
    model_name: Optional[str] = Field(default=None, description="使用するモデル名")


class BatchPredictRequest(BaseModel):
    """バッチ推論リクエスト"""

    images: list[str] = Field(
        ..., max_length=32, description="Base64エンコードされた画像リスト"
    )
    return_confidence: bool = Field(default=True, description="信頼度スコアを返すか")
    model_name: Optional[str] = Field(default=None, description="使用するモデル名")


class TrainRequest(BaseModel):
    """学習リクエスト"""

    dataset_path: str = Field(..., description="データセットパス")
    epochs: int = Field(default=100, ge=1, le=1000, description="エポック数")
    batch_size: int = Field(default=32, ge=1, le=256, description="バッチサイズ")
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1, description="学習率")
    pretrained: bool = Field(default=True, description="事前学習済み重みを使用")
    backbone: str = Field(default="resnet50", description="バックボーンモデル")


class CategoryUpdateRequest(BaseModel):
    """カテゴリ更新リクエスト"""

    cause_categories: Optional[list[dict]] = Field(
        default=None, description="原因カテゴリリスト"
    )
    shape_categories: Optional[list[dict]] = Field(
        default=None, description="形状カテゴリリスト"
    )
    depth_categories: Optional[list[dict]] = Field(
        default=None, description="深さカテゴリリスト"
    )
