import requests
import base64
import json
from pathlib import Path
import sys

def main():
    # API configuration
    api_url = "http://localhost:8000/api/v1/predict"
    
    # Path to a sample image
    # Assuming running from project root
    project_root = Path(__file__).parent.parent
    image_path = project_root / "data/processed/val/images/00000.jpg"
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please ensure you have generated dataset or adjust the image path.")
        return

    print(f"Testing API with image: {image_path}")
    
    # 1. Encode image to base64
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    # 2. Prepare payload
    payload = {
        "image_base64": image_base64,
        "return_confidence": True
    }
    
    # 3. Send POST request
    try:
        print("Sending request to API...")
        response = requests.post(api_url, json=payload)
        
        # 4. Check response
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! API Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nError: API returned status code {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to API at {api_url}")
        print("Make sure the API server is running (python src/api/main.py)")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
