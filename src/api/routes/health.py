"""ヘルスチェックエンドポイント"""

import torch
from fastapi import APIRouter

from src.api.schemas.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """ヘルスチェック"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,  # 実際の実装ではモデルの状態をチェック
        gpu_available=torch.cuda.is_available(),
        version="1.0.0",
    )
