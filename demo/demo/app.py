import gradio as gr
import requests
import base64
import io
import json
import os
from PIL import Image
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

# Configuration
MLFLOW_ENDPOINT = os.getenv("MLFLOW_ENDPOINT", "http://localhost:5000/invocations")
API_KEY = os.getenv("API_KEY", "")

# Sample images paths
SAMPLE_PERSON_PATH = "../sample_images/person_image.jpg"
SAMPLE_CLOTH_PATH = "../sample_images/dress.jpg"

class VirtualTryOnDemo:
    """Modern demo interface for CatVTON virtual try-on application."""
    
    def __init__(self):
        self.setup_interface()
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        if image is None:
            return ""
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def base64_to_image(self, base64_str: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image."""
        try:
            if base64_str.startswith("data:"):
                base64_str = base64_str.split(",", 1)[1]
            
            img_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_data))
            return image
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None
    
    def call_mlflow_endpoint(
        self,
        person_image: Image.Image,
        cloth_image: Image.Image,
        cloth_type: str = "upper",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = -1,
        width: int = 768,
        height: int = 1024,
        return_mask: bool = False
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:
        """
        Call the MLflow endpoint for virtual try-on.
        
        Returns:
            Tuple of (result_image, status_message, mask_image)
        """
        try:
            # Convert images to base64
            person_b64 = self.image_to_base64(person_image)
            cloth_b64 = self.image_to_base64(cloth_image)
            
            if not person_b64 or not cloth_b64:
                return None, "❌ Error: Please provide both person and clothing images", None
            
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
                },
                "params": {
                    "return_mask": return_mask
                }
            }
            
            # Set headers
            headers = {
                "Content-Type": "application/json"
            }
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            
            # Make the request
            response = requests.post(
                MLFLOW_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract result from response
                if "predictions" in result and len(result["predictions"]) > 0:
                    prediction = result["predictions"][0]
                    
                    status = prediction.get("status", "unknown")
                    if status == "ok":
                        result_b64 = prediction.get("result_image")
                        mask_b64 = prediction.get("mask_image") if return_mask else None
                        
                        result_image = self.base64_to_image(result_b64) if result_b64 else None
                        mask_image = self.base64_to_image(mask_b64) if mask_b64 else None
                        
                        return result_image, "✅ Try-on completed successfully!", mask_image
                    else:
                        return None, f"❌ Model error: {status}", None
                else:
                    return None, "❌ No predictions returned from model", None
            else:
                error_msg = f"❌ HTTP {response.status_code}: {response.text}"
                return None, error_msg, None
                
        except requests.exceptions.Timeout:
            return None, "❌ Request timeout. The model might be processing or unavailable.", None
        except requests.exceptions.ConnectionError:
            return None, f"❌ Connection error. Please check if the endpoint is available at {MLFLOW_ENDPOINT}", None
        except Exception as e:
            return None, f"❌ Unexpected error: {str(e)}", None
    
    def process_tryon(
        self,
        person_image: Image.Image,
        cloth_image: Image.Image,
        cloth_type: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
        show_mask: bool,
        progress=gr.Progress()
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]]:
        """
        Process virtual try-on with progress tracking.
        
        Returns:
            Tuple of (result_image, status_message, comparison_image, mask_image)
        """
        if person_image is None or cloth_image is None:
            return None, "❌ Please provide both person and clothing images", None, None
        
        progress(0.1, desc="Validating images...")
        
        # Validate images
        if person_image.mode != "RGB":
            person_image = person_image.convert("RGB")
        if cloth_image.mode != "RGB":
            cloth_image = cloth_image.convert("RGB")
        
        progress(0.3, desc="Sending request to model...")
        
        # Call the endpoint
        result_image, status, mask_image = self.call_mlflow_endpoint(
            person_image=person_image,
            cloth_image=cloth_image,
            cloth_type=cloth_type,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            return_mask=show_mask
        )
        
        progress(0.9, desc="Creating comparison...")
        
        # Create before/after comparison if successful
        comparison_image = None
        if result_image is not None:
            try:
                # Resize images to same height for comparison
                target_height = min(person_image.height, result_image.height, 512)
                
                person_resized = person_image.resize(
                    (int(person_image.width * target_height / person_image.height), target_height),
                    Image.Resampling.LANCZOS
                )
                result_resized = result_image.resize(
                    (int(result_image.width * target_height / result_image.height), target_height),
                    Image.Resampling.LANCZOS
                )
                
                # Create side-by-side comparison
                total_width = person_resized.width + result_resized.width + 20
                comparison_image = Image.new("RGB", (total_width, target_height), (255, 255, 255))
                comparison_image.paste(person_resized, (0, 0))
                comparison_image.paste(result_resized, (person_resized.width + 20, 0))
                
            except Exception as e:
                print(f"Error creating comparison: {e}")
        
        progress(1.0, desc="Complete!")
        
        return result_image, status, comparison_image, mask_image if show_mask else None
    
    def load_sample_images(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Load sample images for demonstration."""
        try:
            person_img = None
            cloth_img = None
            
            if os.path.exists(SAMPLE_PERSON_PATH):
                person_img = Image.open(SAMPLE_PERSON_PATH).convert("RGB")
            
            if os.path.exists(SAMPLE_CLOTH_PATH):
                cloth_img = Image.open(SAMPLE_CLOTH_PATH).convert("RGB")
            
            return person_img, cloth_img
        except Exception as e:
            print(f"Error loading sample images: {e}")
            return None, None
    
    def setup_interface(self):
        """Set up the Gradio interface."""
        
        # Custom CSS for modern styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .title-container {
            text-align: center;
            margin-bottom: 30px;
        }
        .title-container h1 {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .title-container p {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 20px;
        }
        .input-group {
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .output-group {
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=custom_css, title="CatVTON Virtual Try-On Demo") as self.interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="title-container">
                    <h1>🎭 CatVTON Virtual Try-On</h1>
                    <p>AI-powered virtual clothing try-on using diffusion models</p>
                    <p><strong>Upload a person image and clothing item to see the magic happen!</strong></p>
                </div>
                """)
            
            # Main interface
            with gr.Row():
                # Left column - Inputs
                with gr.Column(scale=1, elem_classes="input-group"):
                    gr.HTML("<h3>📸 Input Images</h3>")
                    
                    with gr.Tab("Upload Images"):
                        person_image = gr.Image(
                            label="Person Image",
                            type="pil",
                            height=300,
                            value=None
                        )
                        cloth_image = gr.Image(
                            label="Clothing Image", 
                            type="pil",
                            height=300,
                            value=None
                        )
                    
                    with gr.Tab("Sample Images"):
                        gr.HTML("<p>Click the button below to load sample images for testing:</p>")
                        load_samples_btn = gr.Button("🎯 Load Sample Images", variant="secondary")
                    
                    gr.HTML("<h3>⚙️ Generation Settings</h3>")
                    
                    with gr.Row():
                        cloth_type = gr.Dropdown(
                            choices=["upper", "lower", "overall"],
                            value="upper",
                            label="Clothing Type",
                            info="Type of clothing item"
                        )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Inference Steps",
                            info="More steps = better quality but slower"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale",
                            info="Higher values follow prompt more closely"
                        )
                    
                    with gr.Row():
                        seed = gr.Number(
                            value=-1,
                            label="Seed",
                            info="Use -1 for random, or set a number for reproducible results"
                        )
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        with gr.Row():
                            width = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                value=768,
                                step=64,
                                label="Width"
                            )
                            height = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                value=1024,
                                step=64,
                                label="Height"
                            )
                        
                        show_mask = gr.Checkbox(
                            value=False,
                            label="Show Mask",
                            info="Display the generated mask used for try-on"
                        )
                    
                    # Generate button
                    generate_btn = gr.Button(
                        "✨ Generate Try-On",
                        variant="primary",
                        size="lg"
                    )
                
                # Right column - Outputs
                with gr.Column(scale=1, elem_classes="output-group"):
                    gr.HTML("<h3>🎨 Results</h3>")
                    
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to generate...",
                        interactive=False,
                        show_copy_button=False
                    )
                    
                    with gr.Tab("Try-On Result"):
                        result_image = gr.Image(
                            label="Generated Try-On",
                            type="pil",
                            height=400,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Before/After Comparison"):
                        comparison_image = gr.Image(
                            label="Before vs After",
                            type="pil",
                            height=400,
                            show_download_button=True
                        )
                    
                    with gr.Tab("Mask Visualization"):
                        mask_image = gr.Image(
                            label="Generated Mask",
                            type="pil",
                            height=400,
                            show_download_button=True
                        )
            
            # Footer with information
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e1e5e9;">
                    <h4>📝 How to Use</h4>
                    <p><strong>1.</strong> Upload a clear image of a person (preferably full body or upper body)</p>
                    <p><strong>2.</strong> Upload an image of the clothing item you want to try on</p>
                    <p><strong>3.</strong> Select the appropriate clothing type (upper, lower, or overall)</p>
                    <p><strong>4.</strong> Adjust generation settings if needed</p>
                    <p><strong>5.</strong> Click "Generate Try-On" and wait for the magic! ✨</p>
                    <br>
                    <p><em>⚡ Powered by CatVTON and MLflow</em></p>
                </div>
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.process_tryon,
                inputs=[
                    person_image, cloth_image, cloth_type,
                    num_inference_steps, guidance_scale, seed,
                    width, height, show_mask
                ],
                outputs=[result_image, status_text, comparison_image, mask_image],
                show_progress="full"
            )
            
            load_samples_btn.click(
                fn=self.load_sample_images,
                inputs=[],
                outputs=[person_image, cloth_image]
            )
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        return self.interface.launch(**kwargs)

def main():
    """Main function to run the demo."""
    print("🚀 Starting CatVTON Virtual Try-On Demo...")
    print(f"📡 MLflow Endpoint: {MLFLOW_ENDPOINT}")
    
    demo = VirtualTryOnDemo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main() 