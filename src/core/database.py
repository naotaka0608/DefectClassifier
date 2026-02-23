"""データベース管理モジュール"""

import sqlite3
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
from src.core.constants import DATA_DIR
from src.core.logger import logger

DB_PATH = DATA_DIR / "hantei.db"

from contextlib import closing

class Database:
    """判定履歴を管理するデータベースクラス"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """データベースとテーブルの初期化"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    cause_label TEXT,
                    cause_confidence REAL,
                    shape_label TEXT,
                    shape_confidence REAL,
                    depth_label TEXT,
                    depth_confidence REAL,
                    inference_time_ms REAL,
                    model_version TEXT,
                    is_anomaly INTEGER DEFAULT 0,
                    anomaly_score REAL DEFAULT 1.0,
                    details_json TEXT
                )
            """)
            # 既存のテーブルにカラムがない場合の追加 (migration)
            try:
                cursor.execute("ALTER TABLE classification_history ADD COLUMN is_anomaly INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE classification_history ADD COLUMN anomaly_score REAL DEFAULT 1.0")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE classification_history ADD COLUMN details_json TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def save_result(
        self,
        image_path: str,
        cause: Dict[str, Any],
        shape: Dict[str, Any],
        depth: Dict[str, Any],
        inference_time_ms: float,
        model_version: str = "unknown",
        is_anomaly: bool = False,
        anomaly_score: float = 1.0,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """判定結果を保存"""
        timestamp = datetime.now().isoformat()
        
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classification_history (
                    timestamp, image_path, 
                    cause_label, cause_confidence,
                    shape_label, shape_confidence,
                    depth_label, depth_confidence,
                    inference_time_ms, model_version,
                    is_anomaly, anomaly_score,
                    details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, image_path,
                cause.get("label"), cause.get("confidence"),
                shape.get("label"), shape.get("confidence"),
                depth.get("label"), depth.get("confidence"),
                inference_time_ms, model_version,
                1 if is_anomaly else 0, anomaly_score,
                json.dumps(details) if details else None
            ))
            conn.commit()
            return cursor.lastrowid

    def get_history(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """判定履歴を取得"""
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM classification_history 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_history_item(self, item_id: int):
        """履歴アイテムを削除"""
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM classification_history WHERE id = ?", (item_id,))
            conn.commit()

# シングルトンインスタンス
db = Database()
