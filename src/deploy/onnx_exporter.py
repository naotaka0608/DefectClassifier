
import argparse
from pathlib import Path
import torch
import onnx
from src.models.defect_classifier import DefectClassifier

def export_to_onnx(model_path: Path | str, output_path: Path | str, device: str = "cpu"):
    """
    PyTorchモデルをONNX形式に変換して保存
    
    Args:
        model_path: .pthファイルのパス
        output_path: .onnxファイルの出力パス
        device: 'cpu' or 'cuda'
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    print(f"Loading model from {model_path}...")
    try:
        model = DefectClassifier.load(model_path, device=device)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # ダミー入力の作成
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # 動的軸の設定
    dynamic_axes = {
        "input": {0: "batch_size"},
        "cause": {0: "batch_size"},
        "shape": {0: "batch_size"},
        "depth": {0: "batch_size"},
    }
    
    # 出力名の定義 (Wrapperの返り値の順序に合わせる)
    output_names = ["cause", "shape", "depth"]
    
    from src.core.types import TaskType
    import torch.nn as nn

    class ONNXWrapper(nn.Module):
        """ONNX出力用に辞書をタプルに変換するラッパー"""
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            out = self.model(x)
            # 順序を固定してTensorのTupleを返す
            return (
                out[TaskType.CAUSE],
                out[TaskType.SHAPE],
                out[TaskType.DEPTH]
            )
            
    # Wrap model
    wrapped_model = ONNXWrapper(model)
    
    print(f"Exporting to {output_path}...")
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        
        # Verify
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX export successful!")
        
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DefectClassifier to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .onnx file")
    
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)
