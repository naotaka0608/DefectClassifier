"""動的カテゴリ管理モジュール"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from src.core.logger import logger


@dataclass
class Category:
    """カテゴリ情報"""

    name: str
    description: str
    code: str
    severity: Optional[int] = None


class CategoryManager:
    """動的カテゴリ管理クラス"""

    def __init__(self, config_path: Path | str):
        self.config_path = Path(config_path)
        self._categories: dict[str, list[Category]] = {}
        self._load_categories()

    def _load_categories(self) -> None:
        """設定ファイルからカテゴリを読み込み"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for task_name, items in config["categories"].items():
            self._categories[task_name] = [
                Category(
                    name=item["name"],
                    description=item["description"],
                    code=item["code"],
                    severity=item.get("severity"),
                )
                for item in items
            ]

        logger.info(f"Loaded categories: {self.get_summary()}")

    def reload(self) -> None:
        """カテゴリを再読み込み"""
        self._load_categories()

    def get_categories(self, task_name: str) -> list[str]:
        """タスクのカテゴリ名リストを取得"""
        return [cat.name for cat in self._categories[task_name]]

    def get_category_details(self, task_name: str) -> list[Category]:
        """タスクのカテゴリ詳細リストを取得"""
        return self._categories[task_name]

    def get_category_codes(self, task_name: str) -> list[str]:
        """タスクのカテゴリコードリストを取得"""
        return [cat.code for cat in self._categories[task_name]]

    def get_num_classes(self, task_name: str) -> int:
        """タスクのクラス数を取得"""
        return len(self._categories[task_name])

    def get_all_num_classes(self) -> dict[str, int]:
        """全タスクのクラス数を取得"""
        return {task: len(cats) for task, cats in self._categories.items()}

    def get_summary(self) -> dict:
        """カテゴリサマリーを取得"""
        return {
            task: {"count": len(cats), "names": [c.name for c in cats]}
            for task, cats in self._categories.items()
        }

    def name_to_index(self, task_name: str, category_name: str) -> int:
        """カテゴリ名からインデックスを取得"""
        names = self.get_categories(task_name)
        return names.index(category_name)

    def index_to_name(self, task_name: str, index: int) -> str:
        """インデックスからカテゴリ名を取得"""
        return self._categories[task_name][index].name

    def is_compatible_with_model(self, model_config: dict) -> bool:
        """モデルとの互換性をチェック"""
        return (
            self.get_num_classes("cause") == model_config.get("num_cause_classes")
            and self.get_num_classes("shape") == model_config.get("num_shape_classes")
            and self.get_num_classes("depth") == model_config.get("num_depth_classes")
        )

    def save_categories(self, categories_dict: dict) -> None:
        """カテゴリを設定ファイルに保存"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["categories"] = categories_dict

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        self._load_categories()
