"""推論エンドポイント"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas.request import BatchPredictRequest, PredictRequest
from src.api.schemas.response import (
    BatchPredictResponse,
    ClassificationResultSchema,
    PredictResponse,
)
from src.core.category_manager import CategoryManager
from src.inference.predictor import DefectPredictor

router = APIRouter(prefix="/api/v1", tags=["prediction"])

# グローバル変数（実際の実装では依存性注入を使用）
_predictor: Optional[DefectPredictor] = None
_category_manager: Optional[CategoryManager] = None


def get_predictor() -> DefectPredictor:
    """推論器を取得"""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def set_predictor(predictor: DefectPredictor):
    """推論器を設定"""
    global _predictor
    _predictor = predictor


def set_category_manager(manager: CategoryManager):
    """カテゴリマネージャーを設定"""
    global _category_manager
    _category_manager = manager


@router.post("/predict", response_model=PredictResponse)
async def predict_single(
    request: PredictRequest,
    predictor: DefectPredictor = Depends(get_predictor),
) -> PredictResponse:
    """単一画像の傷分類を実行"""
    start_time = time.perf_counter()

    try:
        result = predictor.predict_from_base64(
            image_base64=request.image_base64,
            return_confidence=request.return_confidence,
        )

        inference_time = (time.perf_counter() - start_time) * 1000

        return PredictResponse(
            success=True,
            cause=ClassificationResultSchema(
                label=result.cause.label,
                confidence=result.cause.confidence,
                class_id=result.cause.class_id,
            ),
            shape=ClassificationResultSchema(
                label=result.shape.label,
                confidence=result.shape.confidence,
                class_id=result.shape.class_id,
            ),
            depth=ClassificationResultSchema(
                label=result.depth.label,
                confidence=result.depth.confidence,
                class_id=result.depth.class_id,
            ),
            inference_time_ms=inference_time,
            model_version=predictor.model_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    predictor: DefectPredictor = Depends(get_predictor),
) -> BatchPredictResponse:
    """バッチ画像の傷分類を実行"""
    start_time = time.perf_counter()

    try:
        # 各画像をデコード
        import base64
        import io

        import numpy as np
        from PIL import Image

        images = []
        for img_b64 in request.images:
            image_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(np.array(image))

        results = predictor.predict_batch(
            images=images,
            return_confidence=request.return_confidence,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        response_results = []
        for result in results:
            response_results.append(
                PredictResponse(
                    success=True,
                    cause=ClassificationResultSchema(
                        label=result.cause.label,
                        confidence=result.cause.confidence,
                        class_id=result.cause.class_id,
                    ),
                    shape=ClassificationResultSchema(
                        label=result.shape.label,
                        confidence=result.shape.confidence,
                        class_id=result.shape.class_id,
                    ),
                    depth=ClassificationResultSchema(
                        label=result.depth.label,
                        confidence=result.depth.confidence,
                        class_id=result.depth.class_id,
                    ),
                    inference_time_ms=total_time / len(results),
                    model_version=predictor.model_version,
                )
            )

        return BatchPredictResponse(
            success=True,
            results=response_results,
            total_inference_time_ms=total_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
