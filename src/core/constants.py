from pathlib import Path

# Base Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
INBOX_DIR = DATA_DIR / "inbox"
CHECKPOINTS_DIR = Path("checkpoints")
CONFIG_DIR = Path("config")

# Training Data
TRAIN_DIR = PROCESSED_DIR / "train"
TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
ANNOTATIONS_FILE = TRAIN_DIR / "annotations.json"

# Configuration
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
CATEGORIES_CONFIG_PATH = CONFIG_DIR / "categories.yaml"

# Models
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
FINAL_MODEL_PATH = CHECKPOINTS_DIR / "final_model.pth"
