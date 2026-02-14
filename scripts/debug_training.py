
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    try:
        from scripts.test_training_run import test_training
        test_training()
    except Exception:
        with open("debug_output.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
