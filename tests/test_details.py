"""詳細な確率分布保存のテスト"""

import unittest
import os
import json
from pathlib import Path
from src.core.database import Database

class TestDetailedHistory(unittest.TestCase):
    def setUp(self):
        self.test_db_path = Path("tests/test_details.db")
        if self.test_db_path.exists():
            os.remove(self.test_db_path)
        self.db = Database(db_path=self.test_db_path)

    def tearDown(self):
        if self.test_db_path.exists():
            os.remove(self.test_db_path)

    def test_save_and_retrieve_details(self):
        # 詳細データ（全候補の確率）
        details = {
            "cause": {"擦り傷": 0.8, "打痕": 0.1, "汚れ": 0.1},
            "shape": {"線状": 0.9, "点状": 0.1},
            "depth": {"浅い": 0.7, "深い": 0.3}
        }
        
        # 保存
        res_id = self.db.save_result(
            image_path="test_details.jpg",
            cause={"label": "擦り傷", "confidence": 0.8},
            shape={"label": "線状", "confidence": 0.9},
            depth={"label": "浅い", "confidence": 0.7},
            inference_time_ms=15.0,
            details=details
        )
        self.assertGreater(res_id, 0)
        
        # 取得
        history = self.db.get_history()
        self.assertEqual(len(history), 1)
        
        # JSONデコードの確認
        retrieved_details_json = history[0]["details_json"]
        self.assertIsNotNone(retrieved_details_json)
        
        retrieved_details = json.loads(retrieved_details_json)
        self.assertEqual(retrieved_details["cause"]["擦り傷"], 0.8)
        self.assertEqual(retrieved_details["depth"]["深い"], 0.3)

if __name__ == "__main__":
    unittest.main()
