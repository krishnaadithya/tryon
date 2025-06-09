#!sudo apt-get update && sudo apt-get install -y libgl1
#%pip install -r ../requirements.txt

import logging
import warnings
import json
import os
import base64
import io
from pathlib import Path
from typing import Any, Dict, Tuple, List
from io import BytesIO

# Third-Party Libraries
import shutil
import torch
import pandas as pd
from PIL import Image

# MLflow for Experiment Tracking and Model Management
import mlflow
from mlflow import MlflowClient
from mlflow.types.schema import Schema, ColSpec
from mlflow.types import ParamSchema, ParamSpec
from mlflow.models import ModelSignature

# Model imports
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from diffusers.image_processor import VaeImageProcessor
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from huggingface_hub import snapshot_download
from diffusers import UNet2DConditionModel

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*diffusion_pytorch_model.safetensors.*")
warnings.filterwarnings("ignore", message=".*unsafe serialization.*")

# Define global experiment and run names
BASE_CHECKPOINT = snapshot_download("runwayml/stable-diffusion-inpainting")
#"booksforcharlie/stable-diffusion-inpainting" #baase checkopoint
EXPERIMENT_NAME = "AI Try On"
ATTENTION_CHECKPOINT = snapshot_download("zhengchong/CatVTON")
#"zhengchong/CatVTON" #attn checkpoint
MODEL_NAME = "TRYON"
RUN_NAME = 'TRYON'
NAME = 'TRYON'

# === Create logger ===
logger = logging.getLogger("deployment-notebook")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", 
                             datefmt="%Y-%m-%d %H:%M:%S") 

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False

logger.info('Starting optimized model deployment...')

## Pipeline Test
pipeline = CatVTONPipeline(
    base_ckpt=BASE_CHECKPOINT,
    attn_ckpt=ATTENTION_CHECKPOINT,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device='cuda'
)

# Load UNet manually with safetensors if available
try:
    unet = UNet2DConditionModel.from_pretrained(
        BASE_CHECKPOINT, 
        subfolder="unet",
        use_safetensors=True
    )
except:
    unet = UNet2DConditionModel.from_pretrained(
        BASE_CHECKPOINT, 
        subfolder="unet",
        use_safetensors=False
    )

# MLflow PyFunc Model Class (lightweight version)
class VirtualTryOnModel(mlflow.pyfunc.PythonModel):
    """Optimized PyFunc wrapper for CatVTON virtual try-on."""
    
    @staticmethod
    def _b64_to_pil(b64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")

    def _pil_to_b64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 PNG string."""
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _preprocess(self, row: pd.Series) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        """Preprocess input data."""
        person = self._b64_to_pil(row["person_image"])
        cloth = self._b64_to_pil(row["cloth_image"])

        meta = {
            "cloth_type": row.get("cloth_type", "upper"),
            "num_inference_steps": int(row.get("num_inference_steps", 20)),  # Reduced default
            "guidance_scale": float(row.get("guidance_scale", 7.5)),
            "seed": int(row.get("seed", -1)),
            "width": int(row.get("width", 512)),   # Reduced default resolution
            "height": int(row.get("height", 768)),
        }

        # Import utils here to avoid loading during model registration
        from utils import resize_and_crop, resize_and_padding
        person = resize_and_crop(person, (meta["width"], meta["height"]))
        cloth = resize_and_padding(cloth, (meta["width"], meta["height"]))

        return person, cloth, meta

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        """Initialize models only when needed."""
        logger.info("Loading CatVTON pipeline...")
        
        # Import heavy dependencies only when loading
        from model.pipeline import CatVTONPipeline
        from model.cloth_masker import AutoMasker
        from diffusers.image_processor import VaeImageProcessor
        from utils import init_weight_dtype
        
        # Get model paths from environment or use defaults
        base_model_id = context.artifacts['base_checkpoint']
        catvton_model_id = context.artifacts['attention_checkpoint']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipeline with model IDs (will download if needed)
        self.pipe = CatVTONPipeline(
            base_ckpt=base_model_id,
            attn_ckpt=catvton_model_id,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype("bf16" if self.device == "cuda" else "fp32"),
            use_tf32=True,
            device=self.device,
        )

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        
        # Download CatVTON repo for masker components
        #from huggingface_hub import snapshot_download
        #attn_repo = snapshot_download(catvton_model_id)
        
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(catvton_model_id, "DensePose"),
            schp_ckpt=os.path.join(catvton_model_id, "SCHP"),
            device=self.device,
        )
        
        logger.info("CatVTON pipeline loaded successfully")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """Run inference."""
        params = params or {}
        return_mask = bool(params.get("return_mask", False))

        outputs = []
        for idx, row in model_input.iterrows():
            try:
                logger.info(f"Processing row {idx + 1}/{len(model_input)}")
                person, cloth, meta = self._preprocess(row)

                # Generate mask
                mask = self.automasker(person, meta["cloth_type"])["mask"]
                mask = self.mask_processor.blur(mask, blur_factor=9)

                # Set up generator
                generator = None
                if meta["seed"] != -1:
                    generator = torch.Generator(device=self.device).manual_seed(meta["seed"])

                # Run inference
                with torch.inference_mode():  # Optimize inference
                    result = self.pipe(
                        image=person,
                        condition_image=cloth,
                        mask=mask,
                        num_inference_steps=meta["num_inference_steps"],
                        guidance_scale=meta["guidance_scale"],
                        generator=generator,
                    )[0]

                output = self._pil_to_b64(result)
                outputs.append({"result_image": output})
                
            except Exception as e:
                logger.exception(f"Inference failed for row {idx}")
                outputs.append({"result_image": "error"})
        
        return pd.DataFrame(outputs)
    
    @classmethod
    def log_model(cls, model_name, source_pipeline=None,demo_folder="../demo"):
        #define the schema for the model
        try:
            # Define lightweight model signature
            input_schema = Schema([
                ColSpec("string", "person_image"),
                ColSpec("string", "cloth_image"),
                ColSpec("string", "cloth_type"),
                ColSpec("long", "num_inference_steps"),
                ColSpec("double", "guidance_scale"),
                ColSpec("long", "seed"),
                ColSpec("long", "width"),
                ColSpec("long", "height"),
            ])
            
            output_schema = Schema([
                ColSpec("string", "result_image"),
                ])

            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            # if source_pipeline:
            #     # os.makedirs(model_name, exist_ok=True)

            #     # Save model and tokenizer properly
            #     source_pipeline.save_pretrained(model_name)
                # source_pipeline.model.save_pretrained(model_name, safe_serialization=True)
                # source_pipeline.tokenizer.save_pretrained(model_name)
                # if hasattr(source_pipeline, "processor"):
                #     source_pipeline.processor.save_pretrained(model_name)
            requirements = [
                "torch>=2.0.0",
                "torchvision",
                "transformers>=4.20.0",
                "diffusers>=0.20.0",
                "huggingface_hub>=0.15.0",
                "Pillow>=9.0.0",
                "accelerate>=0.20.0",
                "pandas",
                "numpy",
                "opencv-python",
                "scipy",
            ]
            artifacts = {
                "base_checkpoint": BASE_CHECKPOINT,
                "attention_checkpoint": ATTENTION_CHECKPOINT,
                "demo": demo_folder,
            }
            
            # Log model via MLflow pyfunc
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=cls(),
                artifacts=artifacts,
                signature=signature,
                pip_requirements=requirements
            )

            # # Only remove the model directory AFTER MLflow has copied it
            # if source_pipeline and os.path.exists(model_name):
            #     shutil.rmtree(model_name)
            logger.info("Logging model to MLflow done successfully")
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {str(e)}")



# Set up MLflow tracking
mlflow.set_tracking_uri('/phoenix/mlflow')

# Create or get experiment
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info(f"Created experiment: {EXPERIMENT_NAME}")
    else:
        logger.info(f"Using existing experiment: {EXPERIMENT_NAME}")
except Exception as e:
    logger.warning(f"Experiment setup issue: {e}")

mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


# Log and register the model (no large artifacts)
logger.info("Starting model registration...")

with mlflow.start_run(run_name= RUN_NAME) as run:
    logger.info(f"Run's Artifact URI: {run.info.artifact_uri}")
    VirtualTryOnModel.log_model(model_name = MODEL_NAME, source_pipeline=None,demo_folder='../demo')
    mlflow.register_model(model_uri = f"runs:/{run.info.run_id}/{MODEL_NAME}", name = NAME)
# with mlflow.start_run(run_name=RUN_NAME) as run:
#     # Log model without heavy artifacts
#     mlflow.pyfunc.log_model(
#         artifact_path="catvton_pyfunc",
#         python_model=VirtualTryOnModel(),
#         pip_requirements=requirements,
#         signature=signature,
#         # No artifacts - models downloaded at runtime
#     )
    
#     # Register the model
#     model_uri = f"runs:/{run.info.run_id}/catvton_pyfunc"
    
#     try:
#         model_version = mlflow.register_model(
#             model_uri=model_uri,
#             name=REGISTERED_MODEL_NAME
#         )
        
#         logger.info(f"Model registered successfully!")
#         logger.info(f"Model name: {REGISTERED_MODEL_NAME}")
#         logger.info(f"Model version: {model_version.version}")
#         logger.info(f"Model URI: {model_uri}")
#         logger.info(f"Run ID: {run.info.run_id}")
        
#     except Exception as e:
#         logger.error(f"Model registration failed: {e}")
#         raise

# logger.info('Optimized model deployment completed successfully!')

# Optional: Test the registered model
def test_registered_model():
    """Test the registered model with sample data."""
    try:
        logger.info("Testing registered model...")
        
        client = mlflow.MlflowClient()
        model_metadata = client.get_latest_versions('TRYON', stages=["None"])
        print('model metadata',model_metadata)
        latest_model_version = model_metadata[0].version
        print('latest model version',latest_model_version, mlflow.models.get_model_info(f"models:/TRYON/{latest_m odel_version}").signature)
        
        # Create proper base64 encoded images
        image_path = "../sample_images/dress.jpg"
        dress_path = "../sample_images/person_image.jpg"
        
        # Proper way to encode images to base64
        def image_to_b64(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        
        person_image_base64 = image_to_b64(image_path)
        dress_image_base64 = image_to_b64(dress_path)
        
        test_data = pd.DataFrame({
            "person_image": [person_image_base64],
            "cloth_image": [dress_image_base64],
            "cloth_type": ["upper"],
            "num_inference_steps": [50],  # Reduced for faster testing
            "guidance_scale": [2.5],
            "seed": [42],
            "width": [768],  # Reduced for faster testing
            "height": [1024],
        })
        
        model = mlflow.pyfunc.load_model(model_uri=f"models:/TRYON/{latest_model_version}")
        # Run prediction
        result = model.predict(test_data)
        image = Image.open(BytesIO(base64.b64decode(result.iloc[0]['result_image'])))
        image.save("result.png")
        
        logger.info("Model test successful!")
        logger.info(f"Result: {result.iloc[0]['result_image'][:50]}...")  # Show first 50 chars of base64
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")

# Uncomment to test after deployment
test_registered_model()