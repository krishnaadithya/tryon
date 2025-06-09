#!/usr/bin/env python3
"""
Standalone CatVTON Demo - Works without MLflow endpoint

This version loads the model directly and provides a local demo interface.
Useful for testing and development without setting up MLflow serving.
"""

import gradio as gr
import base64
import io
import os
import sys
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import torch

# Add parent directory to path to import model components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class StandaloneTryOnDemo:
    """Standalone demo that loads the model directly."""
    
    def __init__(self):
        self.pipeline = None
        self.automasker = None
        self.mask_processor = None
        self.setup_model()
        self.setup_interface()
    
    def setup_model(self):
        """Initialize the CatVTON pipeline and components."""
        try:
            print("🔄 Loading CatVTON model components...")
            
            # Import model components
            from model.cloth_masker import AutoMasker
            from model.pipeline import CatVTONPipeline
            from diffusers.image_processor import VaeImageProcessor
            from utils import init_weight_dtype, resize_and_crop, resize_and_padding
            from huggingface_hub import snapshot_download
            
            # Store utility functions
            self.resize_and_crop = resize_and_crop
            self.resize_and_padding = resize_and_padding
            
            # Download model components
            print("📥 Downloading model weights...")
            MODEL_PERSONAL_NAME = "zhengchong/CatVTON"
            MODEL_NAME = "booksforcharlie/stable-diffusion-inpainting"
            
            repo_path = snapshot_download(repo_id=MODEL_PERSONAL_NAME)
            
            # Initialize pipeline
            print("🚀 Initializing pipeline...")
            self.pipeline = CatVTONPipeline(
                base_ckpt=MODEL_NAME,
                attn_ckpt=repo_path,
                attn_ckpt_version="mix",
                weight_dtype=init_weight_dtype("bf16"),
                use_tf32=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Initialize mask processor
            self.mask_processor = VaeImageProcessor(
                vae_scale_factor=8, 
                do_normalize=False, 
                do_binarize=True, 
                do_convert_grayscale=True
            )
            
            # Initialize automasker
            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please ensure you have the required dependencies installed.")
            print("Run: pip install -r ../requirements.txt")
            self.pipeline = None
    
    def process_tryon_local(
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
        Process virtual try-on locally without API calls.
        
        Returns:
            Tuple of (result_image, status_message, comparison_image, mask_image)
        """
        if self.pipeline is None:
            return None, "❌ Model not loaded. Please check the setup.", None, None
        
        if person_image is None or cloth_image is None:
            return None, "❌ Please provide both person and clothing images", None, None
        
        try:
            progress(0.1, desc="Preprocessing images...")
            
            # Validate and convert images
            if person_image.mode != "RGB":
                person_image = person_image.convert("RGB")
            if cloth_image.mode != "RGB":
                cloth_image = cloth_image.convert("RGB")
            
            # Resize images
            person_processed = self.resize_and_crop(person_image, (width, height))
            cloth_processed = self.resize_and_padding(cloth_image, (width, height))
            
            progress(0.3, desc="Generating mask...")
            
            # Generate mask
            mask = self.automasker(person_processed, cloth_type)["mask"]
            mask = self.mask_processor.blur(mask, blur_factor=9)
            
            progress(0.5, desc="Running diffusion pipeline...")
            
            # Set up generator for reproducibility
            generator = None
            if seed != -1:
                generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
            
            # Run the pipeline
            result_image = self.pipeline(
                image=person_processed,
                condition_image=cloth_processed,
                mask=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )[0]
            
            progress(0.9, desc="Creating comparison...")
            
            # Create before/after comparison
            comparison_image = None
            if result_image is not None:
                try:
                    # Resize images for comparison
                    target_height = min(person_processed.height, result_image.height, 512)
                    
                    person_resized = person_processed.resize(
                        (int(person_processed.width * target_height / person_processed.height), target_height),
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
            
            return (
                result_image,
                "✅ Try-on completed successfully!",
                comparison_image,
                mask if show_mask else None
            )
            
        except Exception as e:
            error_msg = f"❌ Error during processing: {str(e)}"
            print(error_msg)
            return None, error_msg, None, None
    
    def load_sample_images(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Load sample images for demonstration."""
        try:
            sample_person_path = "../sample_images/person_image.jpg"
            sample_cloth_path = "../sample_images/dress.jpg"
            
            person_img = None
            cloth_img = None
            
            if os.path.exists(sample_person_path):
                person_img = Image.open(sample_person_path).convert("RGB")
            
            if os.path.exists(sample_cloth_path):
                cloth_img = Image.open(sample_cloth_path).convert("RGB")
            
            return person_img, cloth_img
        except Exception as e:
            print(f"Error loading sample images: {e}")
            return None, None
    
    def setup_interface(self):
        """Set up the Gradio interface."""
        
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
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        """
        
        with gr.Blocks(css=custom_css, title="CatVTON Standalone Demo") as self.interface:
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div class="title-container">
                    <h1>🎭 CatVTON Standalone Demo</h1>
                    <p>AI-powered virtual clothing try-on running locally</p>
                    <p><strong>No API required - model runs directly on your machine!</strong></p>
                </div>
                """)
            
            # Status indicator
            if self.pipeline is None:
                gr.HTML("""
                <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 10px; margin: 20px 0;">
                    ❌ <strong>Model Not Loaded</strong><br>
                    Please ensure you have installed all dependencies and the model files are accessible.
                </div>
                """)
            else:
                device_info = "GPU" if torch.cuda.is_available() else "CPU"
                gr.HTML(f"""
                <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 10px; margin: 20px 0;">
                    ✅ <strong>Model Loaded Successfully</strong><br>
                    Running on: {device_info} | Device: {self.pipeline.device if self.pipeline else 'Unknown'}
                </div>
                """)
            
            # Main interface
            with gr.Row():
                # Left column - Inputs
                with gr.Column(scale=1):
                    gr.HTML("<h3>📸 Input Images</h3>")
                    
                    with gr.Tab("Upload Images"):
                        person_image = gr.Image(
                            label="Person Image",
                            type="pil",
                            height=300
                        )
                        cloth_image = gr.Image(
                            label="Clothing Image", 
                            type="pil",
                            height=300
                        )
                    
                    with gr.Tab("Sample Images"):
                        gr.HTML("<p>Click to load sample images:</p>")
                        load_samples_btn = gr.Button("🎯 Load Sample Images", variant="secondary")
                    
                    gr.HTML("<h3>⚙️ Generation Settings</h3>")
                    
                    cloth_type = gr.Dropdown(
                        choices=["upper", "lower", "overall"],
                        value="upper",
                        label="Clothing Type"
                    )
                    
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=50,
                            step=5,
                            label="Inference Steps"
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    
                    with gr.Row():
                        seed = gr.Number(
                            value=-1,
                            label="Seed (-1 for random)"
                        )
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        with gr.Row():
                            width = gr.Slider(512, 1024, value=768, step=64, label="Width")
                            height = gr.Slider(512, 1024, value=1024, step=64, label="Height")
                        
                        show_mask = gr.Checkbox(value=False, label="Show Mask")
                    
                    generate_btn = gr.Button(
                        "✨ Generate Try-On",
                        variant="primary",
                        size="lg",
                        interactive=(self.pipeline is not None)
                    )
                
                # Right column - Outputs
                with gr.Column(scale=1):
                    gr.HTML("<h3>🎨 Results</h3>")
                    
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to generate..." if self.pipeline else "Model not loaded",
                        interactive=False
                    )
                    
                    with gr.Tab("Try-On Result"):
                        result_image = gr.Image(
                            label="Generated Try-On",
                            type="pil",
                            height=400
                        )
                    
                    with gr.Tab("Before/After"):
                        comparison_image = gr.Image(
                            label="Before vs After",
                            type="pil",
                            height=400
                        )
                    
                    with gr.Tab("Mask"):
                        mask_image = gr.Image(
                            label="Generated Mask",
                            type="pil",
                            height=400
                        )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e1e5e9;">
                <p><strong>Standalone Demo</strong> - Runs the model locally without API dependencies</p>
                <p><em>⚡ Powered by CatVTON</em></p>
            </div>
            """)
            
            # Event handlers
            if self.pipeline is not None:
                generate_btn.click(
                    fn=self.process_tryon_local,
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
    """Main function to run the standalone demo."""
    print("🚀 Starting CatVTON Standalone Demo...")
    print("📝 This version runs the model locally without requiring MLflow")
    
    demo = StandaloneTryOnDemo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflicts
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main() 