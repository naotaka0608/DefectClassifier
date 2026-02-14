
import sys
import base64
import json
import requests
from pathlib import Path
from PIL import Image
import io
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.constants import TRAIN_DIR

def test_batch_inference():
    print("Starting batch inference test...")
    
    # API Endpoint
    api_url = "http://localhost:8000/api/v1/predict/batch"
    
    # Use images from the generated dataset as "external" images
    # Select 3 images
    image_paths = sorted((TRAIN_DIR / "images").glob("*.jpg"))[:3]
    
    if not image_paths:
        print("No images found for testing.")
        return

    print(f"Testing with {len(image_paths)} images:")
    for p in image_paths:
        print(f" - {p.name}")

    # Prepare payload
    images_base64 = []
    for p in image_paths:
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
            images_base64.append(img_b64)
            
    payload = {
        "images": images_base64,
        "return_confidence": True
    }
    
    try:
        print("\nSending batch request...")
        # Note: Ensure uvicorn is running! 
        # For this test script, we assume the user or a separate process has started the API.
        # If not, this request will fail.
        
        # However, for a self-contained test, we might want to import the router or predictor directly
        # to avoid needing a running server. 
        # Let's try to use the Predictor directly first to verify the logic, 
        # as starting a server in a script is complex.
        
        # Method A: Direct Predictor Usage (Integration Test)
        from src.inference.predictor import DefectPredictor
        from src.core.constants import BEST_MODEL_PATH
        from src.core.config import load_config
        from src.core.category_manager import CategoryManager
        from src.core.constants import CONFIG_DIR
        
        print("\n[Method A] Testing DefectPredictor directly...")
        
        # Load resources
        # model_config is loaded inside DefectPredictor via DefectClassifier.load
        category_manager = CategoryManager(CONFIG_DIR / "categories.yaml")
        
        predictor = DefectPredictor(
            model_path=BEST_MODEL_PATH,
            category_manager=category_manager
        )
        
        # Helper to convert b64 to structured inputs
        numpy_images = []
        for b64_str in images_base64:
            img_data = base64.b64decode(b64_str)
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            numpy_images.append(np.array(pil_image))
            
        # Run batch prediction
        results = predictor.predict_batch(numpy_images)
        
        # Convert dataclasses to dicts for JSON serialization
        import dataclasses
        results_dict = [dataclasses.asdict(r) for r in results]
        
        # Verify JSON serializability
        json_output = json.dumps(results_dict, indent=2, ensure_ascii=False)
        print("\nBatch Prediction Result (JSON):")
        print(json_output)
        
        # Check structure
        assert isinstance(results, list)
        assert len(results) == len(image_paths)
        for res_dict in results_dict:
            assert "cause" in res_dict
            assert "shape" in res_dict
            assert "depth" in res_dict
            
        print("\n[Method A] Direct prediction successful!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch_inference()
