"""異常検知・レポート生成の動作テスト"""

import unittest
from pathlib import Path
from src.utils.report_generator import ReportGenerator
from src.inference.anomaly_detector import AnomalyDetector
import torch

class TestAdvancedFeatures(unittest.TestCase):
    def test_anomaly_detector(self):
        detector = AnomalyDetector(threshold=0.4)
        
        # 正常ケース (高確信度)
        logits_normal = {
            "cause": torch.tensor([[0.1, 0.9]]), # 90%
            "shape": torch.tensor([[0.05, 0.95]]), # 95%
            "depth": torch.tensor([[0.1, 0.9]]) # 90%
        }
        is_anomaly, score = detector.detect(logits_normal)
        self.assertFalse(is_anomaly)
        self.assertGreater(score, 0.4)
        
        # 異常ケース (低確信度)
        logits_anomaly = {
            "cause": torch.tensor([[0.5, 0.5]]), # 50% (境界付近)
            "shape": torch.tensor([[0.5, 0.5]]),
            "depth": torch.tensor([[0.1, 0.1, 0.8]]) # これは高いが、他が低い場合
        }
        # この例だと 1/(1+1) = 0.5 なので、0.4より大きい。
        # もっと低くする
        logits_unknown = {
            "cause": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]) # 均等なら 0.2
        }
        is_anomaly, score = detector.detect(logits_unknown)
        self.assertTrue(is_anomaly)
        self.assertLess(score, 0.4)

    def test_report_generation(self):
        gen = ReportGenerator()
        sample_result = {
            "id": 999,
            "timestamp": "2026-02-24 05:00:00",
            "image_path": "tests/sample.jpg", # setUpで作成が必要
            "cause_label": "擦り傷",
            "cause_confidence": 0.95,
            "shape_label": "線状",
            "shape_confidence": 0.88,
            "depth_label": "浅い",
            "depth_confidence": 0.92,
            "inference_time_ms": 12.5,
            "model_version": "test_v1",
            "is_anomaly": 0
        }
        
        # サンプル画像作成
        img_path = Path("tests/sample_report.jpg")
        from PIL import Image
        Image.new("RGB", (100, 100), color="blue").save(img_path)
        sample_result["image_path"] = str(img_path)
        
        output_pdf = Path("tests/test_report.pdf")
        pdf_buffer = gen.generate(sample_result, output_path=output_pdf)
        
        self.assertTrue(output_pdf.exists())
        self.assertGreater(output_pdf.stat().st_size, 0)
        
        # クリーンアップ
        if output_pdf.exists():
            output_pdf.unlink()
        if img_path.exists():
            img_path.unlink()

if __name__ == "__main__":
    unittest.main()
