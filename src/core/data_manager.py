import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.constants import ANNOTATIONS_FILE, TRAIN_IMAGES_DIR, DATA_DIR


class DataManager:
    """データセット管理クラス"""

    def __init__(self, annotation_file: Path | str = ANNOTATIONS_FILE):
        self.annotation_file = Path(annotation_file)

    def load_annotations(self) -> list[dict]:
        """アノテーションファイルを読み込む"""
        if not self.annotation_file.exists():
            return []

        try:
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = []
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and "samples" in data:
                samples = data["samples"]

            # マイグレーション
            for item in samples:
                # image_path補完 (新形式 -> 旧形式的な補完)
                # 注: image_pathは本来絶対パスか相対パスか揺れているが、
                # train/dataset.py等ではこれを使っている箇所があるかもしれない
                if "image_path" not in item and "file_name" in item:
                    item["image_path"] = f"processed/train/images/{item['file_name']}"
                
                # file_name補完 (Legacy対応: image_path -> file_name)
                if "file_name" not in item and "image_path" in item:
                    item["file_name"] = Path(item["image_path"]).name

            return samples
        except Exception:
            return []

    def save_annotations(self, annotations: list[dict]) -> None:
        """アノテーションを保存"""
        # 親ディレクトリ作成
        self.annotation_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.annotation_file, "w", encoding="utf-8") as f:
            json.dump({"samples": annotations}, f, ensure_ascii=False, indent=2)

    def add_sample(
        self,
        file_name: str,
        cause: str,
        shape: str,
        depth: str,
        source: str = "upload",
        original_name: Optional[str] = None,
    ) -> None:
        """
        サンプルをデータセットに追加
        NOTE: 画像ファイル自体の移動/保存は呼び出し元で行うこと
        """
        annotations = self.load_annotations()
        
        # 相対パス計算
        # file_name は TRAIN_IMAGES_DIR 直下にあると仮定
        image_path = TRAIN_IMAGES_DIR / file_name
        rel_path = image_path.relative_to(DATA_DIR)
        
        new_sample = {
            "image_path": str(rel_path).replace("\\", "/"), # JSONは/"区切り推奨
            "file_name": file_name,
            "cause": cause,
            "shape": shape,
            "depth": depth,
            "source": source,
            "added_at": datetime.now().isoformat(),
        }
        
        if original_name:
            new_sample["original_name"] = original_name
            
        annotations.append(new_sample)
        self.save_annotations(annotations)

    def import_file(
        self,
        src_path: Path | str,
        cause: str,
        shape: str,
        depth: str,
        source: str = "upload",
    ) -> Path:
        """
        ファイルをデータセットディレクトリにインポートし、メタデータを追加
        
        Args:
            src_path: インポート元のファイルパス
            cause, shape, depth: ラベル
            source: データソース
            
        Returns:
            dst_path: インポート先のファイルパス
        """
        import shutil
        
        src_path = Path(src_path)
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
            
        # ファイル名衝突回避
        file_name = src_path.name
        dst_path = TRAIN_IMAGES_DIR / file_name
        
        if dst_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{timestamp}_{src_path.name}"
            dst_path = TRAIN_IMAGES_DIR / file_name
            
        # ファイル移動
        shutil.move(str(src_path), str(dst_path))
        
        # メタデータ登録
        self.add_sample(
            file_name=file_name,
            cause=cause,
            shape=shape,
            depth=depth,
            source=source,
            original_name=src_path.name
        )
        
        return dst_path
