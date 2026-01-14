"""
GLM-Image Worker for TensorBase Serverless

A production-ready GLM-Image worker for text-to-image and image-to-image generation,
integrating with the TensorBase serverless infrastructure.

GLM-Image is a hybrid autoregressive + diffusion image generation model with:
- 9B autoregressive generator + 7B diffusion decoder
- Excellent text rendering in images
- Text-to-image and image-to-image capabilities
- High-quality semantic understanding

Based on: https://huggingface.co/zai-org/GLM-Image
"""

import os
import io
import asyncio
import time
import logging
import base64
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Configuration from environment
MODEL_ID = os.getenv("MODEL_ID", "zai-org/GLM-Image")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "50"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "1.5"))
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))

# TensorBase integration
CALLBACK_URL = os.getenv("TENSORBASE_CALLBACK_URL", "")
WORKER_ID = os.getenv("TENSORBASE_WORKER_ID", "")
ENDPOINT_ID = os.getenv("TENSORBASE_ENDPOINT_ID", "")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glm-image-worker")

# Global pipeline
pipe = None

# ============================================
# Pydantic Models
# ============================================

class TextToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the image to generate")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt for guidance")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height (must be divisible by 32)")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width (must be divisible by 32)")
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=1.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    num_images: int = Field(default=1, ge=1, le=4)

class ImageToImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the edit/transformation")
    image: str  # Base64 encoded image or URL
    negative_prompt: Optional[str] = None
    height: Optional[int] = Field(default=None, description="Output height (must be divisible by 32)")
    width: Optional[int] = Field(default=None, description="Output width (must be divisible by 32)")
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=1.5, ge=1.0, le=20.0)
    seed: Optional[int] = None

class MultiImageRequest(BaseModel):
    """For multi-subject consistency and multi-image-to-image generation."""
    prompt: str
    images: List[str]  # List of base64 encoded images or URLs
    negative_prompt: Optional[str] = None
    height: int = Field(default=1024)
    width: int = Field(default=1024)
    num_inference_steps: int = Field(default=50)
    guidance_scale: float = Field(default=1.5)
    seed: Optional[int] = None

class ImageGenerationResponse(BaseModel):
    images: List[str]  # Base64 encoded images
    seed: int
    width: int
    height: int

class TensorBaseJobRequest(BaseModel):
    id: str
    input: Dict[str, Any]

# ============================================
# Pipeline Setup
# ============================================

def get_torch_dtype():
    """Get PyTorch dtype from config."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(TORCH_DTYPE, torch.bfloat16)

def ensure_divisible_by_32(value: int) -> int:
    """Ensure dimension is divisible by 32 as required by GLM-Image."""
    return (value // 32) * 32

async def initialize_pipeline():
    """Initialize the GLM-Image pipeline."""
    global pipe
    
    logger.info(f"Loading GLM-Image model: {MODEL_ID}")
    logger.info(f"Torch dtype: {TORCH_DTYPE}")
    logger.info("Model will be auto-downloaded from HuggingFace if not cached...")
    
    loop = asyncio.get_event_loop()
    
    def _load_pipeline():
        from diffusers.pipelines.glm_image import GlmImagePipeline
        
        return GlmImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=get_torch_dtype(),
            device_map="cuda",
            token=HF_TOKEN,
        )
    
    pipe = await loop.run_in_executor(None, _load_pipeline)
    
    logger.info("GLM-Image pipeline loaded successfully!")
    
    # Notify TensorBase that worker is ready
    await notify_worker_ready()

async def notify_worker_ready():
    """Notify TensorBase orchestrator that this worker is ready."""
    if not CALLBACK_URL or not WORKER_ID:
        logger.info("No callback URL configured, skipping ready notification")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CALLBACK_URL}/internal/worker/ready",
                json={
                    "workerId": WORKER_ID,
                    "endpointId": ENDPOINT_ID,
                },
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info("Notified orchestrator: worker ready")
            else:
                logger.warning(f"Failed to notify ready: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to notify worker ready: {e}")

async def notify_job_complete(job_id: str, output: Dict[str, Any], error: Optional[str] = None):
    """Notify TensorBase orchestrator that a job is complete."""
    if not CALLBACK_URL:
        return
    
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "jobId": job_id,
                "workerId": WORKER_ID,
            }
            if error:
                payload["error"] = error
            else:
                payload["output"] = output
            
            await client.post(
                f"{CALLBACK_URL}/internal/job/complete",
                json=payload,
                timeout=30.0
            )
    except Exception as e:
        logger.error(f"Failed to notify job complete: {e}")

async def send_heartbeat():
    """Send periodic heartbeat to orchestrator."""
    if not CALLBACK_URL or not WORKER_ID:
        return
    
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{CALLBACK_URL}/internal/worker/heartbeat",
                    json={"workerId": WORKER_ID},
                    timeout=10.0
                )
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
        await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup/shutdown."""
    # Startup
    await initialize_pipeline()
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())
    
    yield
    
    # Shutdown
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="GLM-Image Worker",
    description="Text-to-image and image-to-image generation server using GLM-Image",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================
# Helper Functions
# ============================================

async def load_image_from_source(source: str) -> Image.Image:
    """Load image from base64 string or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        async with httpx.AsyncClient() as client:
            response = await client.get(source, timeout=60.0)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
    else:
        if source.startswith("data:"):
            _, source = source.split(",", 1)
        image_bytes = base64.b64decode(source)
        image = Image.open(io.BytesIO(image_bytes))
    
    return image.convert("RGB")

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def generate_text_to_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: Optional[int] = None,
    num_images: int = 1,
) -> tuple[List[Image.Image], int]:
    """Generate images from text prompt using GLM-Image."""
    if pipe is None:
        raise RuntimeError("Pipeline not loaded")
    
    # Ensure dimensions are divisible by 32
    height = ensure_divisible_by_32(height)
    width = ensure_divisible_by_32(width)
    
    # Set seed
    if seed is None:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    loop = asyncio.get_event_loop()
    
    def _generate():
        images = []
        for _ in range(num_images):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            images.append(result.images[0])
        return images
    
    images = await loop.run_in_executor(None, _generate)
    return images, seed

async def generate_image_to_image(
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: Optional[int] = None,
) -> tuple[Image.Image, int, int, int]:
    """Generate image from input image + prompt using GLM-Image."""
    if pipe is None:
        raise RuntimeError("Pipeline not loaded")
    
    # Use input image dimensions if not specified
    if height is None:
        height = image.height
    if width is None:
        width = image.width
    
    # Ensure dimensions are divisible by 32
    height = ensure_divisible_by_32(height)
    width = ensure_divisible_by_32(width)
    
    # Set seed
    if seed is None:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    loop = asyncio.get_event_loop()
    
    def _generate():
        result = pipe(
            prompt=prompt,
            image=[image],  # GLM-Image expects a list for image input
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
    
    output_image = await loop.run_in_executor(None, _generate)
    return output_image, seed, width, height

async def generate_multi_image(
    images: List[Image.Image],
    prompt: str,
    negative_prompt: Optional[str] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.5,
    seed: Optional[int] = None,
) -> tuple[Image.Image, int]:
    """Generate image from multiple input images + prompt (multi-subject consistency)."""
    if pipe is None:
        raise RuntimeError("Pipeline not loaded")
    
    # Ensure dimensions are divisible by 32
    height = ensure_divisible_by_32(height)
    width = ensure_divisible_by_32(width)
    
    # Set seed
    if seed is None:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    loop = asyncio.get_event_loop()
    
    def _generate():
        result = pipe(
            prompt=prompt,
            image=images,  # Multiple images for multi-subject generation
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
    
    output_image = await loop.run_in_executor(None, _generate)
    return output_image, seed

# ============================================
# API Endpoints
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipe is not None else "loading",
        "model": MODEL_ID,
        "worker_id": WORKER_ID,
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": "glm-image",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zai-org",
            "capabilities": [
                "text-to-image",
                "image-to-image",
                "image-editing",
                "style-transfer",
                "multi-subject-consistency",
                "text-rendering",
            ],
            "specs": {
                "parameters": "16B (9B AR + 7B Decoder)",
                "architecture": "hybrid autoregressive + diffusion",
            }
        }]
    }

@app.post("/v1/images/generations")
async def create_image(request: TextToImageRequest):
    """Generate images from text prompt (OpenAI-compatible)."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        images, seed = await generate_text_to_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            num_images=request.num_images,
        )
        
        images_base64 = [image_to_base64(img) for img in images]
        
        return ImageGenerationResponse(
            images=images_base64,
            seed=seed,
            width=ensure_divisible_by_32(request.width),
            height=ensure_divisible_by_32(request.height),
        )
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/edits")
async def edit_image(request: ImageToImageRequest):
    """Edit/transform an image based on prompt (OpenAI-compatible)."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load input image
        image = await load_image_from_source(request.image)
        
        output_image, seed, width, height = await generate_image_to_image(
            image=image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        image_base64 = image_to_base64(output_image)
        
        return ImageGenerationResponse(
            images=[image_base64],
            seed=seed,
            width=width,
            height=height,
        )
        
    except Exception as e:
        logger.error(f"Image editing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/images/multi")
async def multi_image_generation(request: MultiImageRequest):
    """Generate image from multiple input images (multi-subject consistency)."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load all input images
        images = []
        for img_source in request.images:
            img = await load_image_from_source(img_source)
            images.append(img)
        
        output_image, seed = await generate_multi_image(
            images=images,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        
        image_base64 = image_to_base64(output_image)
        
        return ImageGenerationResponse(
            images=[image_base64],
            seed=seed,
            width=ensure_divisible_by_32(request.width),
            height=ensure_divisible_by_32(request.height),
        )
        
    except Exception as e:
        logger.error(f"Multi-image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
async def run_job(request: TensorBaseJobRequest):
    """TensorBase-compatible job execution endpoint."""
    job_id = request.id
    input_data = request.input
    
    try:
        task = input_data.get("task", "text-to-image")
        
        if task == "text-to-image":
            images, seed = await generate_text_to_image(
                prompt=input_data.get("prompt", ""),
                negative_prompt=input_data.get("negative_prompt"),
                height=input_data.get("height", DEFAULT_HEIGHT),
                width=input_data.get("width", DEFAULT_WIDTH),
                num_inference_steps=input_data.get("num_inference_steps", DEFAULT_STEPS),
                guidance_scale=input_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
                seed=input_data.get("seed"),
                num_images=input_data.get("num_images", 1),
            )
            
            output = {
                "images": [image_to_base64(img) for img in images],
                "seed": seed,
                "width": ensure_divisible_by_32(input_data.get("width", DEFAULT_WIDTH)),
                "height": ensure_divisible_by_32(input_data.get("height", DEFAULT_HEIGHT)),
            }
            
        elif task == "image-to-image":
            image = await load_image_from_source(input_data.get("image", ""))
            
            output_image, seed, width, height = await generate_image_to_image(
                image=image,
                prompt=input_data.get("prompt", ""),
                negative_prompt=input_data.get("negative_prompt"),
                height=input_data.get("height"),
                width=input_data.get("width"),
                num_inference_steps=input_data.get("num_inference_steps", DEFAULT_STEPS),
                guidance_scale=input_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
                seed=input_data.get("seed"),
            )
            
            output = {
                "images": [image_to_base64(output_image)],
                "seed": seed,
                "width": width,
                "height": height,
            }
            
        elif task == "multi-image":
            images = []
            for img_source in input_data.get("images", []):
                img = await load_image_from_source(img_source)
                images.append(img)
            
            output_image, seed = await generate_multi_image(
                images=images,
                prompt=input_data.get("prompt", ""),
                negative_prompt=input_data.get("negative_prompt"),
                height=input_data.get("height", DEFAULT_HEIGHT),
                width=input_data.get("width", DEFAULT_WIDTH),
                num_inference_steps=input_data.get("num_inference_steps", DEFAULT_STEPS),
                guidance_scale=input_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
                seed=input_data.get("seed"),
            )
            
            output = {
                "images": [image_to_base64(output_image)],
                "seed": seed,
                "width": ensure_divisible_by_32(input_data.get("width", DEFAULT_WIDTH)),
                "height": ensure_divisible_by_32(input_data.get("height", DEFAULT_HEIGHT)),
            }
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        await notify_job_complete(job_id, output)
        
        return {"status": "completed", "output": output}
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await notify_job_complete(job_id, {}, error=str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)}
        )

# ============================================
# Run with Uvicorn
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
