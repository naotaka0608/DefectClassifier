import requests
import base64
import random
from pathlib import Path
import time

def main():
    # API configuration
    api_url = "http://localhost:8000/api/v1/predict"
    
    # Path to images
    project_root = Path(__file__).parent.parent
    images_dir = project_root / "data/processed/val/images"
    
    # Get all jpg images
    image_paths = sorted(list(images_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f"Error: No images found in {images_dir}")
        return

    # Select 5 distinct images
    selected_images = image_paths[:5]
    if len(image_paths) < 5:
        selected_images = image_paths
        
    print(f"Sending {len(selected_images)} distinct images to API...")
    
    models = ["best_model", "final_model"]
    
    for i, image_path in enumerate(selected_images):
        print(f"\n[{i+1}/{len(selected_images)}] Sending {image_path.name}...")
        
        # 1. Encode image to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Alternate models
        model_name = models[i % len(models)]
        
        payload = {
            "image_base64": image_base64,
            "return_confidence": True,
            "model_name": model_name
        }
        
        try:
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                print(f"Success! Model: {model_name}")
            else:
                print(f"Error: API returned status code {response.status_code}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            
        # Wait a bit to ensure timestamps are different (though uuid should handle it)
        time.sleep(1)

if __name__ == "__main__":
    main()
