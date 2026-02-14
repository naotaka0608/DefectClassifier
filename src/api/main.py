"""FastAPI メインアプリケーション"""

from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.logger import logger

from src.api.routes import categories, health, predict


import yaml
from src.core.category_manager import CategoryManager
from src.core.constants import CHECKPOINTS_DIR, CONFIG_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH
from src.inference.predictor import DefectPredictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル"""
    logger.info("Starting Hantei API...")
    
    model_config_path = CONFIG_DIR / "model_config.yaml"
    category_config_path = CONFIG_DIR / "categories.yaml"
    
    try:
        # カテゴリマネージャー初期化
        logger.info("Loading category manager...")
        category_manager = CategoryManager(category_config_path)
        app.state.category_manager = category_manager
        
        # モデル設定読み込み
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        app.state.config = model_config
            
        # モデル初期化
        logger.info("Loading model...")
        model_path = BEST_MODEL_PATH
        if not model_path.exists():
            logger.warning(f"Default model not found at {model_path}. Trying final_model.pth")
            model_path = FINAL_MODEL_PATH
            
        if model_path.exists():
            predictor = DefectPredictor(
                model_path=model_path,
                category_manager=category_manager
            )
            app.state.predictor = predictor
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.error("No model found! API will return errors for predictions.")
            
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        
    yield
    logger.info("Shutting down Hantei API...")



# FastAPIアプリケーション
app = FastAPI(
    title="傷分類API",
    description="製品検品用の傷分類機械学習API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(categories.router)


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "傷分類API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
