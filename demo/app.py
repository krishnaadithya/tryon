import os
import base64
import io
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import requests
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT", "http://localhost:8000/")
API_KEY = os.getenv("API_KEY", "")
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class VirtualTryOnService:
    """Service class for handling virtual try-on operations."""
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        if image is None:
            return ""
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image."""
        try:
            if base64_str.startswith("data:"):
                base64_str = base64_str.split(",", 1)[1]
            
            img_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_data))
            return image
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return None
    
    @staticmethod
    def call_mlflow_endpoint(
        person_image: Image.Image,
        cloth_image: Image.Image,
        cloth_type: str = "upper",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1,
        width: int = 512,
        height: int = 768
    ) -> Tuple[Optional[Image.Image], str]:
        """Call the MLflow endpoint for virtual try-on."""
        try:
            # Convert images to base64
            person_b64 = VirtualTryOnService.image_to_base64(person_image)
            cloth_b64 = VirtualTryOnService.image_to_base64(cloth_image)
            
            if not person_b64 or not cloth_b64:
                return None, "Error: Failed to process images"
            
            # Prepare the request payload
            payload = {
                "dataframe_split": {
                    "columns": [
                        "person_image", "cloth_image", "cloth_type",
                        "num_inference_steps", "guidance_scale", "seed",
                        "width", "height"
                    ],
                    "data": [[
                        person_b64, cloth_b64, cloth_type,
                        num_inference_steps, guidance_scale, seed,
                        width, height
                    ]]
                }
            }
            
            # Set headers
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            
            logger.info(f"Sending request to {MLFLOW_ENDPOINT}")
            
            # Make the request
            response = requests.post(
                MLFLOW_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received response: {type(result)}")
                
                # Handle different response formats
                if "predictions" in result and len(result["predictions"]) > 0:
                    # MLflow batch format
                    prediction = result["predictions"][0]
                    if isinstance(prediction, dict) and "result_image" in prediction:
                        result_b64 = prediction["result_image"]
                    else:
                        result_b64 = prediction
                elif isinstance(result, list) and len(result) > 0:
                    # Direct list format
                    prediction = result[0]
                    if isinstance(prediction, dict) and "result_image" in prediction:
                        result_b64 = prediction["result_image"]
                    else:
                        result_b64 = prediction
                else:
                    return None, "Unexpected response format from model"
                
                if result_b64 and result_b64 != "error":
                    result_image = VirtualTryOnService.base64_to_image(result_b64)
                    if result_image:
                        return result_image, "Success"
                    else:
                        return None, "Failed to decode result image"
                else:
                    return None, "Model returned error"
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None, error_msg
                
        except requests.exceptions.Timeout:
            return None, "Request timeout. The model might be processing or unavailable."
        except requests.exceptions.ConnectionError:
            return None, f"Connection error. Please check if the endpoint is available at {MLFLOW_ENDPOINT}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None, f"Unexpected error: {str(e)}"

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/api/tryon', methods=['POST'])
def tryon():
    """Handle virtual try-on requests."""
    try:
        # Get uploaded files
        person_file = request.files.get('person_image')
        cloth_file = request.files.get('cloth_image')
        
        if not person_file or not cloth_file:
            return jsonify({
                'error': 'Both person_image and cloth_image are required'
            }), 400
        
        # Get parameters
        cloth_type = request.form.get('cloth_type', 'upper')
        num_inference_steps = int(request.form.get('num_inference_steps', 20))
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        seed = int(request.form.get('seed', -1))
        width = int(request.form.get('width', 512))
        height = int(request.form.get('height', 768))
        
        # Process images
        person_image = Image.open(person_file.stream).convert('RGB')
        cloth_image = Image.open(cloth_file.stream).convert('RGB')
        
        # Save input images for display
        person_filename = f"person_{person_file.filename}"
        cloth_filename = f"cloth_{cloth_file.filename}"
        
        person_path = os.path.join(UPLOAD_FOLDER, person_filename)
        cloth_path = os.path.join(UPLOAD_FOLDER, cloth_filename)
        
        person_image.save(person_path)
        cloth_image.save(cloth_path)
        
        # Call the virtual try-on service
        result_image, status = VirtualTryOnService.call_mlflow_endpoint(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type=cloth_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height
        )
        
        if result_image is not None:
            # Save result image
            result_filename = f"result_{person_file.filename.split('.')[0]}_{cloth_file.filename}"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            result_image.save(result_path)
            
            return jsonify({
                'success': True,
                'status': status,
                'person_url': f'/uploads/{person_filename}',
                'cloth_url': f'/uploads/{cloth_filename}',
                'result_url': f'/results/{result_filename}'
            })
        else:
            return jsonify({
                'success': False,
                'error': status,
                'person_url': f'/uploads/{person_filename}',
                'cloth_url': f'/uploads/{cloth_filename}'
            })
            
    except Exception as e:
        logger.error(f"Error in tryon endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860) 