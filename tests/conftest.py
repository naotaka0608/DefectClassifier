import pytest
from pathlib import Path
import shutil
import tempfile
import yaml
import json
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.category_manager import CategoryManager
from src.core.data_manager import DataManager


@pytest.fixture
def temp_dir():
    """一時ディレクトリを作成"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_categories_config(temp_dir):
    """ダミーのカテゴリ設定ファイル"""
    config = {
        "categories": {
            "cause": [
                {"name": "scratches", "description": "傷", "code": "SCRATCH"},
                {"name": "dents", "description": "打痕", "code": "DENT"}
            ],
            "shape": [
                {"name": "circle", "description": "円形", "code": "CIRCLE"},
                {"name": "square", "description": "四角", "code": "SQUARE"}
            ],
            "depth": [
                {"name": "shallow", "description": "浅い", "code": "SHALLOW"},
                {"name": "deep", "description": "深い", "code": "DEEP"}
            ]
        }
    }
    config_path = temp_dir / "categories.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def mock_annotations_file(temp_dir):
    """ダミーのアノテーションファイル"""
    data = [
        {
            "id": "1",
            "metadata": {"file_name": "test1.jpg"},
            "cause": "scratches",
            "shape": "circle",
            "depth": "shallow",
            "file_name": "test1.jpg"
        },
        {
            "id": "2",
            "metadata": {"file_name": "test2.jpg"},
            "cause": "dents",
            "shape": "square",
            "depth": "deep",
            "file_name": "test2.jpg"
        }
    ]
    file_path = temp_dir / "annotations.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def category_manager(mock_categories_config):
    return CategoryManager(mock_categories_config)

@pytest.fixture
def data_manager(mock_annotations_file):
    return DataManager(mock_annotations_file)

@pytest.fixture
def mock_predictor():
    from unittest.mock import MagicMock
    predictor = MagicMock()
    predictor.model_version = "v1.0.0"
    
    # モックの予測結果
    from src.inference.predictor import PredictionResult, ClassificationResult
    result = PredictionResult(
        cause=ClassificationResult("scratches", 0.95, 0),
        shape=ClassificationResult("circle", 0.92, 0),
        depth=ClassificationResult("shallow", 0.88, 0)
    )
    predictor.predict_from_base64.return_value = result
    
    # バッチ推論の結果
    predictor.predict_batch.return_value = [result, result]
    
    return predictor

@pytest.fixture
def test_client(category_manager, mock_predictor):
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def mock_lifespan(app):
        app.state.category_manager = category_manager
        app.state.predictor = mock_predictor
        app.state.config = {"api": {"save_received_images": False}}
        yield

    app.router.lifespan_context = mock_lifespan
    
    with TestClient(app) as client:
        yield client
