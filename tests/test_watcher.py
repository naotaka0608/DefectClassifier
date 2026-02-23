"""Watcher 動作テスト"""

import time
import shutil
import unittest
from pathlib import Path
from src.services.watcher import DefectWatcher
from src.core.database import db

class TestWatcher(unittest.TestCase):
    def setUp(self):
        self.monitor_dir = Path("tests/monitor_test")
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # サンプルの1x1白画像
        self.sample_img = Path("tests/sample.jpg")
        from PIL import Image
        Image.new("RGB", (1, 1), color="white").save(self.sample_img)
        
        self.watcher = DefectWatcher(self.monitor_dir)

    def tearDown(self):
        self.watcher.stop()
        if self.monitor_dir.exists():
            shutil.rmtree(self.monitor_dir)
        if self.sample_img.exists():
            self.sample_img.unlink()

    def test_watcher_notification(self):
        # 履歴の現在数
        initial_count = len(db.get_history())
        
        # 監視開始
        self.watcher.start()
        
        # ファイルを監視フォルダにコピー
        target_path = self.monitor_dir / "test_capture.jpg"
        shutil.copy(self.sample_img, target_path)
        
        # 処理待ち (watchdog のイベント発生 + 推論時間)
        max_wait = 5
        count = 0
        while count < max_wait:
            time.sleep(1)
            current_count = len(db.get_history())
            if current_count > initial_count:
                break
            count += 1
            
        self.assertGreater(len(db.get_history()), initial_count, "Database should have a new entry")
        
        # 最新の履歴がこのテストのものか確認
        latest = db.get_history(limit=1)[0]
        self.assertIn("test_capture.jpg", latest["image_path"])

if __name__ == "__main__":
    unittest.main()
