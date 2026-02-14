import sys
from pathlib import Path
from loguru import logger

def setup_logger():
    """集中管理用ロガーの設定"""
    # 既存のハンドラを全て削除（重複を防ぐ）
    logger.remove()

    # コンソール出力の設定
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # ログディレクトリの作成
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # ファイル出力の設定（日次ローテーション）
    logger.add(
        log_dir / "hantei_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
        # 各ログエントリの末尾に改行を追加
        backtrace=True,
        diagnose=True,
    )

    return logger

# デフォルトインスタンスの提供
# 各モジュールでは `from src.core.logger import logger` でインポートして使用する
setup_logger()
# 他のモジュールからインポートして使えるように、設定済みのloggerをグローバルに出す
__all__ = ["logger"]
