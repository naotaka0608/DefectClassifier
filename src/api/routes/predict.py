"""推論エンドポイント"""

import time
import uuid
import json
import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from src.core.logger import logger

from src.api.schemas.request import BatchPredictRequest, PredictRequest
from src.api.schemas.response import (
    BatchPredictResponse,
    ClassificationResultSchema,
    PredictResponse,
)
from src.core.category_manager import CategoryManager
from src.core.constants import CHECKPOINTS_DIR, INBOX_DIR
from src.inference.predictor import DefectPredictor

router = APIRouter(prefix="/api/v1", tags=["prediction"])


def _get_predictor(request: Request) -> DefectPredictor:
    """app.state から推論器を取得"""
    predictor: Optional[DefectPredictor] = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def _get_config(request: Request) -> Optional[dict]:
    """app.state から設定を取得"""
    return getattr(request.app.state, "config", None)


def _resolve_model(model_name: str, predictor: DefectPredictor) -> None:
    """モデル名を解決してリロード"""
    model_path = CHECKPOINTS_DIR / model_name
    if not model_path.exists():
        model_path_with_ext = CHECKPOINTS_DIR / f"{model_name}.pth"
        if model_path_with_ext.exists():
            model_path = model_path_with_ext
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found in checkpoints"
            )

    try:
        predictor.reload_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def _save_inference_log(image: Image.Image, result: PredictResponse, config: Optional[dict]):
    """推論ログ（画像と結果）を保存"""
    if config is None or not config.get("api", {}).get("save_received_images", False):
        return

    try:
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{timestamp}_{request_id[:8]}"
        save_dir = INBOX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # 画像保存
        image_path = save_dir / f"{filename_base}.jpg"
        image.save(image_path, quality=95)

        # メタデータ保存
        json_path = save_dir / f"{filename_base}.json"
        result_dict = result.model_dump()
        
        metadata = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "prediction": result_dict
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved inference log: {filename_base}")
            
    except Exception as e:
        logger.error(f"Failed to save inference log: {e}")


def _to_predict_response(
    result,
    inference_time_ms: float,
    model_version: str,
) -> PredictResponse:
    """PredictionResult を PredictResponse に変換"""
    return PredictResponse(
        success=True,
        cause=ClassificationResultSchema(
            label=result.cause.label,
            confidence=result.cause.confidence,
            class_id=result.cause.class_id,
            probabilities=result.cause.probabilities,
        ),
        shape=ClassificationResultSchema(
            label=result.shape.label,
            confidence=result.shape.confidence,
            class_id=result.shape.class_id,
            probabilities=result.shape.probabilities,
        ),
        depth=ClassificationResultSchema(
            label=result.depth.label,
            confidence=result.depth.confidence,
            class_id=result.depth.class_id,
            probabilities=result.depth.probabilities,
        ),
        inference_time_ms=inference_time_ms,
        model_version=model_version,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_single(
    request: PredictRequest,
    raw_request: Request,
) -> PredictResponse:
    """単一画像の傷分類を実行"""
    predictor = _get_predictor(raw_request)
    config = _get_config(raw_request)

    if request.model_name:
        _resolve_model(request.model_name, predictor)

    start_time = time.perf_counter()

    try:
        result = predictor.predict_from_base64(
            image_base64=request.image_base64,
            return_confidence=request.return_confidence,
        )

        inference_time = (time.perf_counter() - start_time) * 1000
        response = _to_predict_response(result, inference_time, predictor.model_version)

        # ログ保存
        try:
            image_bytes = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            _save_inference_log(image, response, config)
        except Exception as e:
            logger.warning(f"Failed to prepare image for logging: {e}")

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    raw_request: Request,
) -> BatchPredictResponse:
    """バッチ画像の傷分類を実行"""
    predictor = _get_predictor(raw_request)
    config = _get_config(raw_request)

    if request.model_name:
        _resolve_model(request.model_name, predictor)

    start_time = time.perf_counter()

    try:
        pil_images = []
        images = []
        for img_b64 in request.images:
            image_bytes = base64.b64decode(img_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(image)
            images.append(np.array(image))

        results = predictor.predict_batch(
            images=images,
            return_confidence=request.return_confidence,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        response_results = []
        for i, result in enumerate(results):
            response = _to_predict_response(
                result, total_time / len(results), predictor.model_version
            )
            response_results.append(response)
            
            # ログ保存
            try:
                _save_inference_log(pil_images[i], response, config)
            except Exception as e:
                logger.warning(f"Failed to save log for batch item {i}: {e}")

        return BatchPredictResponse(
            success=True,
            results=response_results,
            total_inference_time_ms=total_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
