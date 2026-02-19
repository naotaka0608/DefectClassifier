
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

sys.path.append(".")
from src.inference.predictor import DefectPredictor
from src.core.constants import CHECKPOINTS_DIR
from src.core.category_manager import CategoryManager
from src.core.config import DEFAULT_CATEGORIES_CONFIG

# Suppress warnings
warnings.filterwarnings("ignore")

print("Loading model for demo...")
try:
    model_path = CHECKPOINTS_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        sys.exit(1)
        
    category_manager = CategoryManager(DEFAULT_CATEGORIES_CONFIG)
    predictor = DefectPredictor(model_path, category_manager=category_manager)
    print("Model loaded successfully.")

    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        print(f"Samples directory not found: {samples_dir}")
        sys.exit(1)

    print("\n--- Sample Prediction Results ---")
    for img_path in sorted(samples_dir.glob("*.jpg")):
        print(f"\nImage: {img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            
            # Predict
            result = predictor.predict(img_np)
            
            # result structure: cause, shape, depth (PredictionResult objects)
            # wait, Predictor returns PredictionResult which has cause: ClassificationResult
            # ClassificationResult has label, confidence
            
            print(f"  [Cause] {result.cause.label:<10} (Conf: {result.cause.confidence:.2f})")
            print(f"  [Shape] {result.shape.label:<10} (Conf: {result.shape.confidence:.2f})")
            print(f"  [Depth] {result.depth.label:<10} (Conf: {result.depth.confidence:.2f})")
            
        except Exception as e:
            print(f"  Error processing image: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n---------------------------------")
    print("Demo completed.")

except Exception as e:
    print(f"Demo failed: {e}")
    import traceback
    traceback.print_exc()
