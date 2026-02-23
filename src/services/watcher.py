"""フォルダ監視サービス"""

import time
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import numpy as np
from src.core.logger import logger
from src.core.database import db
from src.inference.predictor import DefectPredictor
from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG
from src.core.constants import BEST_MODEL_PATH, HISTORY_DIR

class DefectHandler(FileSystemEventHandler):
    """ファイル作成イベントを処理するハンドラ"""

    def __init__(self, predictor: DefectPredictor):
        self.predictor = predictor

    def _wait_for_file_ready(self, file_path: Path, timeout: float = 5.0) -> bool:
        """ファイルが完全に書き込まれ、開けるようになるまで待機"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 読み取りモードで開けるか試行
                with open(file_path, "rb") as f:
                    # 数バイト読んでみて整合性を確認 (任意)
                    f.read(10)
                return True
            except (IOError, OSError):
                time.sleep(0.2)
        return False

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            if self._wait_for_file_ready(file_path):
                self._process_image(file_path)
            else:
                logger.error(f"Timeout waiting for file to be ready: {file_path}")

    def _process_image(self, file_path: Path):
        try:
            logger.info(f"Auto-processing new image: {file_path.name}")
            
            # 画像読み込み
            image = Image.open(file_path).convert("RGB")
            image_np = np.array(image)
            
            # 推論
            start_time = time.perf_counter()
            result = self.predictor.predict(image_np)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # 履歴保存 (画像も history にコピーするのが安全)
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            history_path = HISTORY_DIR / f"auto_{file_path.name}"
            # 同名ファイルがある場合はリネーム
            if history_path.exists():
                history_path = HISTORY_DIR / f"auto_{int(time.time())}_{file_path.name}"
            
            image.save(history_path, quality=95)
            
            db.save_result(
                image_path=str(history_path),
                cause={"label": result.cause.label, "confidence": result.cause.confidence},
                shape={"label": result.shape.label, "confidence": result.shape.confidence},
                depth={"label": result.depth.label, "confidence": result.depth.confidence},
                inference_time_ms=inference_time,
                model_version=self.predictor.model_version,
                is_anomaly=result.is_anomaly,
                anomaly_score=result.anomaly_score,
                details={
                    "cause": result.cause.probabilities,
                    "shape": result.shape.probabilities,
                    "depth": result.depth.probabilities
                }
            )
            logger.info(f"Auto-classification successful for {file_path.name}: {result.cause.label}")
            
        except Exception as e:
            logger.error(f"Failed to auto-process {file_path.name}: {e}")

class DefectWatcher:
    """フォルダ監視サービス本体"""

    def __init__(self, watch_dir: str | Path):
        self.watch_dir = Path(watch_dir)
        self.observer = None
        self.predictor = None

    def start(self):
        """監視を開始"""
        if self.observer and self.observer.is_alive():
            logger.warning("Watcher is already running")
            return

        # フォルダがなければ作成
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # 推論器の準備
        category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
        self.predictor = DefectPredictor(model_path=BEST_MODEL_PATH, category_manager=category_manager)

        event_handler = DefectHandler(self.predictor)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        logger.info(f"Started monitoring directory: {self.watch_dir}")

    def stop(self):
        """監視を停止"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped monitoring")

if __name__ == "__main__":
    # テスト用
    import sys
    watch_path = sys.argv[1] if len(sys.argv) > 1 else "./data/monitor"
    watcher = DefectWatcher(watch_path)
    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
