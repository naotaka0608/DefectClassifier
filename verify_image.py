
import base64
import io
from PIL import Image

def generate_and_verify():
    # Generate 1x1 white PNG
    img = Image.new('RGB', (1, 1), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    # Verify
    try:
        decoded_bytes = base64.b64decode(base64_str)
        verify_buffer = io.BytesIO(decoded_bytes)
        img_verify = Image.open(verify_buffer)
        img_verify.verify()
        print("Verification successful!")
        
        # Write to file
        with open('valid_base64.txt', 'w') as f:
            f.write(base64_str)
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    generate_and_verify()
