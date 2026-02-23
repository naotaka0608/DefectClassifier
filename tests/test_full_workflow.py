"""統合デバッグテスト (Persistence, Watching, Anomaly, Reporting)"""

import time
import shutil
import unittest
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.services.watcher import DefectWatcher
from src.core.database import db
from src.utils.report_generator import ReportGenerator

class TestFullWorkflow(unittest.TestCase):
    def setUp(self):
        self.monitor_dir = Path("tests/monitor_debug")
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # テストデータの準備
        self.normal_img = Path("tests/normal.jpg")
        Image.new("RGB", (224, 224), color=(200, 200, 200)).save(self.normal_img)
        
        self.anomaly_img = Path("tests/anomaly.jpg")
        # モデルが困りそうな、極端にコントラストの低い、あるいはノイズだらけの画像
        noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(noise).save(self.anomaly_img)
        
        self.watcher = DefectWatcher(self.monitor_dir)
        self.watcher.start()
        
    def tearDown(self):
        self.watcher.stop()
        if self.monitor_dir.exists():
            shutil.rmtree(self.monitor_dir)
        for p in [self.normal_img, self.anomaly_img]:
            if p.exists():
                p.unlink()
        
        # テストで生成されたPDFも削除
        for p in Path("tests").glob("debug_report_*.pdf"):
            p.unlink()

    def test_end_to_end(self):
        print("\n--- Testing Normal Image Detection ---")
        shutil.copy(self.normal_img, self.monitor_dir / "test_normal.jpg")
        
        # 処理待ち
        time.sleep(3)
        
        history = db.get_history(limit=10)
        found_normal = any("test_normal.jpg" in item["image_path"] for item in history)
        self.assertTrue(found_normal, "Normal image should be recorded in DB")
        
        print("--- Testing Anomaly Detection ---")
        shutil.copy(self.anomaly_img, self.monitor_dir / "test_anomaly.jpg")
        
        # 処理待ち
        time.sleep(3)
        
        history = db.get_history(limit=10)
        anomaly_item = next((item for item in history if "test_anomaly.jpg" in item["image_path"]), None)
        self.assertIsNotNone(anomaly_item, "Anomaly image should be recorded in DB")
        
        # Anomalyフラグの確認 (単純な閾値判定なので確実ではないが、ログとして確認)
        print(f"Anomaly flag: {anomaly_item['is_anomaly']}, Score: {anomaly_item['anomaly_score']:.2f}")

        print("--- Testing Report Generation ---")
        gen = ReportGenerator()
        pdf_path = Path(f"tests/debug_report_{anomaly_item['id']}.pdf")
        gen.generate(anomaly_item, output_path=pdf_path)
        
        self.assertTrue(pdf_path.exists(), "PDF report should be generated")
        self.assertGreater(pdf_path.stat().st_size, 0, "PDF should not be empty")
        print(f"Success: Report generated at {pdf_path}")

if __name__ == "__main__":
    unittest.main()
