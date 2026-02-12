"""Streamlit UI起動スクリプト"""

import subprocess
import sys
from pathlib import Path

# プロジェクトルート
project_root = Path(__file__).parent.parent


def main():
    """UIを起動"""
    app_path = project_root / "src" / "ui" / "app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            "8501",
            "--server.address",
            "localhost",
        ],
        cwd=project_root,
    )


if __name__ == "__main__":
    main()
