"""Database 永続化テスト"""

import os
import unittest
from pathlib import Path
from src.core.database import Database

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.test_db_path = Path("tests/test_hantei.db")
        if self.test_db_path.exists():
            os.remove(self.test_db_path)
        self.db = Database(db_path=self.test_db_path)

    def tearDown(self):
        if self.test_db_path.exists():
            os.remove(self.test_db_path)

    def test_save_and_get_history(self):
        # データ保存
        res_id = self.db.save_result(
            image_path="test.jpg",
            cause={"label": "擦り傷", "confidence": 0.9},
            shape={"label": "線状", "confidence": 0.8},
            depth={"label": "表層", "confidence": 0.7},
            inference_time_ms=10.5,
            model_version="test_v1"
        )
        self.assertGreater(res_id, 0)

        # データ取得
        history = self.db.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["cause_label"], "擦り傷")
        self.assertEqual(history[0]["model_version"], "test_v1")

    def test_delete_history(self):
        res_id = self.db.save_result(
            "test.jpg", {"label": "A", "confidence": 0.5}, {"label": "B", "confidence": 0.5}, {"label": "C", "confidence": 0.5}, 1.0
        )
        self.db.delete_history_item(res_id)
        history = self.db.get_history()
        self.assertEqual(len(history), 0)

if __name__ == "__main__":
    unittest.main()
