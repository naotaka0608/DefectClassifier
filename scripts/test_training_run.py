"""学習パイプラインの動作テスト用スクリプト"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.runner import train_model
from src.core.config import load_config

import yaml

def test_training():
    print("Starting training test...")
    
    # テスト用の設定ファイルを作成（エポック数を減らす）
    config_path = project_root / "config/model_config.yaml"
    
    # 既存設定読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # エポック数を1に変更
    original_epochs = config["training"]["epochs"]
    config["training"]["epochs"] = 1
    
    # 一時保存
    temp_config_path = project_root / "config/test_model_config.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    try:
        def progress_callback(metrics):
            print(f"Epoch {metrics['epoch']}/{metrics['total_epochs']} - "
                  f"Train Loss: {metrics['train_loss']:.4f}, "
                  f"Val Loss: {metrics['val_loss']:.4f}")
        
        history = train_model(config_path=temp_config_path, progress_callback=progress_callback)
        print("Training completed successfully!")
        print(f"History keys: {history.keys()}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 設定を戻すわけではないが、一時ファイルは削除する
        if temp_config_path.exists():
            temp_config_path.unlink()

if __name__ == "__main__":
    test_training()
