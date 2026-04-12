import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    StableDiffusionInpaintPipeline = None

class ControlNetInpainter:
    """Part 3 Generative Inpainting for Keyframes using Stable Diffusion."""

    def __init__(self, weights_dir="models/stable-diffusion-inpainting"):
        self.weights_dir = Path(weights_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def _load_model(self):
        if self.pipeline is not None:
            return

        if StableDiffusionInpaintPipeline is None:
            raise ImportError("Please install diffusers, transformers, and accelerate to use Generative Inpainting.")

        os.makedirs(self.weights_dir, exist_ok=True)
        # Using ModelScope instead of HuggingFace Hub to bypass network issues
        model_id = "benjamin-paine/stable-diffusion-v1-5-inpainting"
        print(f"Downloading/Loading '{model_id}' from ModelScope into {self.weights_dir}...")
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(model_id, cache_dir=str(self.weights_dir))
            
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                local_files_only=False
            ).to(self.device)
            self.pipeline.safety_checker = None
        except ImportError:
            print("WARNING: modelscope is not installed. Please `pip install modelscope`.")
            print("Skipping Generative Inpainting for now.")
            self.pipeline = None
            return
        
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass

    def inpaint(self, frames, masks, keyframe_indices=None, prompt="a natural background, highly detailed, photorealistic"):
        """
        Generate background for selected keyframes using SD Inpainting.
        """
        self._load_model()
        
        if self.pipeline is None:
            # Fallback visually if SD fails to load (just returning original frames for placeholder)
            print("[SD Inpainting] Skipping generation, returning original keyframes.")
            if keyframe_indices is None:
                keyframe_indices = [len(frames) // 2]
            return {idx: np.array(frames[idx]) for idx in keyframe_indices}
            
        frames = [f if isinstance(f, np.ndarray) else np.array(f) for f in frames]
        masks = [m if isinstance(m, np.ndarray) else np.array(m) for m in masks]
        
        if keyframe_indices is None:
            keyframe_indices = [len(frames) // 2]
            
        results = {}
        for idx in keyframe_indices:
            frame_rgb = frames[idx]
            mask_bin = masks[idx]
            
            mask_bin = (mask_bin > 0).astype(np.uint8) * 255
                
            image_pil = Image.fromarray(frame_rgb)
            mask_pil = Image.fromarray(mask_bin)
            
            print(f"[SD Inpainting] Generating keyframe {idx} with prompt: '{prompt}'...")
            
            generator = torch.manual_seed(42)
            with torch.autocast(self.device):
                image = self.pipeline(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
            
            results[idx] = np.array(image)
            
        return results