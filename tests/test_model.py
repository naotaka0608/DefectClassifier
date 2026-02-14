
import pytest
import torch
import tempfile
from pathlib import Path
from src.models.defect_classifier import DefectClassifier

def test_defect_classifier_initialization():
    """DefectClassifierがtask_configを使って正しく初期化されるかテスト"""
    task_config = {
        "task1": 2,
        "task2": 5
    }
    # resnet50を使用 (BackboneFactoryでサポートされているため)
    model = DefectClassifier(
        task_config=task_config,
        backbone_name="resnet50",
        pretrained=False
    )
    
    assert model.task_config == task_config
    assert "task1" in model.multi_task_head.heads
    assert "task2" in model.multi_task_head.heads
    # resnet50のfeature_dimは2048
    # shared_features=512 -> head_hidden_features=256 -> out_features
    assert model.multi_task_head.heads["task1"].head[3].out_features == 2
    assert model.multi_task_head.heads["task2"].head[3].out_features == 5

def test_defect_classifier_forward():
    """Forwardパスのテスト"""
    task_config = {"task1": 2}
    model = DefectClassifier(
        task_config=task_config,
        backbone_name="resnet50",
        pretrained=False
    )
    
    # ダミー入力
    x = torch.randn(1, 3, 224, 224)
    # resnet50はメモリを食うのでno_gradで
    with torch.no_grad():
        outputs = model(x)
    
    assert isinstance(outputs, dict)
    assert "task1" in outputs
    assert outputs["task1"].shape == (1, 2)

def test_defect_classifier_save_load():
    """モデルの保存と読み込みテスト"""
    task_config = {"cause": 3, "shape": 2}
    model = DefectClassifier(
        task_config=task_config,
        backbone_name="resnet50",
        pretrained=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model.pth"
        model.save(path)
        
        loaded_model = DefectClassifier.load(path)
        
        assert loaded_model.task_config == task_config
        assert "cause" in loaded_model.multi_task_head.heads
        assert "shape" in loaded_model.multi_task_head.heads

def test_legacy_config_loading():
    """旧形式のConfigからの読み込みテスト"""
    from src.core.types import TaskType
    # TaskTypeの値が文字列であることを前提
    
    legacy_config = {
        "backbone": "resnet50",
        "pretrained": False,
        "num_cause_classes": 10,
        "num_shape_classes": 5,
        "num_depth_classes": 2
    }
    
    model = DefectClassifier.from_config(legacy_config)
    
    # TaskType.CAUSE などの値は "cause", "shape", "depth"
    expected_config = {
        TaskType.CAUSE: 10,
        TaskType.SHAPE: 5,
        TaskType.DEPTH: 2
    }
    
    assert model.task_config == expected_config
    assert TaskType.CAUSE in model.multi_task_head.heads
