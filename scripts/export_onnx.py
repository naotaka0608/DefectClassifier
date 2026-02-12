"""ONNX変換スクリプト（C#連携用）"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.models.defect_classifier import DefectClassifier


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """PyTorchモデルをONNX形式にエクスポート"""
    print(f"Loading model from {checkpoint_path}...")
    model = DefectClassifier.load(checkpoint_path, device="cpu")
    model.eval()

    # ダミー入力
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["cause", "shape", "depth"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "cause": {0: "batch_size"},
            "shape": {0: "batch_size"},
            "depth": {0: "batch_size"},
        },
    )

    print(f"Model exported to {output_path}")

    # 検証
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except ImportError:
        print("Install 'onnx' package for model validation")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
