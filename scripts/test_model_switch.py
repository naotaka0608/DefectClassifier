import requests
import base64
import json
from pathlib import Path
import sys

def main():
    # API configuration
    api_url = "http://localhost:8000/api/v1/predict"
    
    # Path to a sample image
    project_root = Path(__file__).parent.parent
    image_path = project_root / "data/processed/val/images/00000.jpg"
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Encode image to base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    # List of models to test
    # Requires actual model files in checkpoints/ directory
    # For testing, we might use the same model name if only one exists, just to verify logic doesn't crash
    models = ["best_model", "final_model"] 
    
    for model_name in models:
        print(f"\n--- Testing with model: {model_name} ---")
        
        payload = {
            "image_base64": image_base64,
            "return_confidence": True,
            "model_name": model_name
        }
        
        try:
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("Success!")
                print(f"Model Version Used: {result.get('model_version')}")
                # print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
