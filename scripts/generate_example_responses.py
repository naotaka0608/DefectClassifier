import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.api.schemas.response import (
    BatchPredictResponse,
    PredictResponse,
    ClassificationResultSchema,
)

def create_dummy_result(index: int) -> PredictResponse:
    return PredictResponse(
        success=True,
        cause=ClassificationResultSchema(
            label="scratches",
            confidence=0.95,
            class_id=0
        ),
        shape=ClassificationResultSchema(
            label="circle",
            confidence=0.88,
            class_id=1
        ),
        depth=ClassificationResultSchema(
            label="deep",
            confidence=0.75,
            class_id=2
        ),
        inference_time_ms=15.5,
        model_version="v1.0.0"
    )

def generate_batch_response(num_images: int):
    results = [create_dummy_result(i) for i in range(num_images)]
    
    response = BatchPredictResponse(
        success=True,
        results=results,
        total_inference_time_ms=15.5 * num_images + 5.0 # overhead
    )
    
    print(f"\n--- API Response for {num_images} image(s) ---")
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

def main():
    print("Generating example JSON responses for API documentation...")
    
    with open("api_response_examples.json", "w", encoding="utf-8") as f:
        # Redirect stdout to file
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            generate_batch_response(1)
            generate_batch_response(5)
            generate_batch_response(10)
        finally:
            sys.stdout = original_stdout
            
    print("Output written to api_response_examples.json")

if __name__ == "__main__":
    main()
