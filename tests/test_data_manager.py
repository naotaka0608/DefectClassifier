
import pytest
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.core.data_manager import DataManager

@pytest.fixture
def temp_data_dir(tmp_path):
    # Setup structured data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "processed" / "train" / "images").mkdir(parents=True)
    return data_dir

@pytest.fixture
def data_manager(temp_data_dir):
    # Mock CONFIG_DIR and DATA_DIR in constants?
    # DataManager uses DATA_DIR from constants.
    # We need to patch constants or instantiate DataManager with path (if supported).
    # DataManager.__init__ doesn't take paths, it uses constants.
    # So we patch constants.
    with patch("src.core.data_manager.DATA_DIR", temp_data_dir), \
         patch("src.core.data_manager.TRAIN_IMAGES_DIR", temp_data_dir / "processed/train/images"):
         
        dm = DataManager(annotation_file=temp_data_dir / "processed/train/annotations.json")
        yield dm

def test_import_file(data_manager, temp_data_dir):
    # Create valid source file
    src_file = temp_data_dir / "test_import.jpg"
    src_file.write_text("dummy content")
    
    # Import
    dst_path = data_manager.import_file(
        src_path=src_file,
        cause="kiz",
        shape="maru",
        depth="asai"
    )
    
    # Verify file moved
    assert not src_file.exists()
    assert dst_path.exists()
    assert dst_path.name == "test_import.jpg"
    
    # Verify plotting
    annotations = data_manager.load_annotations()
    assert len(annotations) == 1
    assert annotations[0]["file_name"] == "test_import.jpg"
    assert annotations[0]["cause"] == "kiz"

def test_migration_legacy_list(data_manager, temp_data_dir):
    # Prepare legacy annotation file (list format, missing file_name)
    legacy_data = [
        {
            "id": "001",
            "image_path": "processed/train/images/legacy.jpg",
            "cause": "kiz"
        }
    ]
    
    annotation_file = temp_data_dir / "processed/train/annotations.json"
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)
        
    # Load
    annotations = data_manager.load_annotations()
    
    # Verify migration
    assert len(annotations) == 1
    assert "file_name" in annotations[0]
    assert annotations[0]["file_name"] == "legacy.jpg"

def test_migration_legacy_dict(data_manager, temp_data_dir):
    # Prepare legacy annotation file (dict format, samples missing file_name)
    legacy_data = {
        "samples": [
            {
                "id": "002",
                "image_path": "processed/train/images/legacy_dict.jpg",
                "cause": "kiz"
            }
        ]
    }
    
    annotation_file = temp_data_dir / "processed/train/annotations.json"
    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)
        
    # Load
    annotations = data_manager.load_annotations()
    
    # Verify migration
    assert len(annotations) == 1
    assert "file_name" in annotations[0]
    assert annotations[0]["file_name"] == "legacy_dict.jpg"
