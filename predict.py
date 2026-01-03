# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import Sam3Model, Sam3Processor

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use bfloat16 on GPU if available, else float16 on GPU, float32 on CPU
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32  # CPU requires float32 for stability
        
        # Download weights if they don't exist
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)
        
        print(f"Loading model from {MODEL_PATH} to {self.device} with {self.dtype}...")
        self.model = Sam3Model.from_pretrained(MODEL_PATH).to(self.device, dtype=self.dtype).eval()
        self.processor = Sam3Processor.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image file"),
        prompt: str = Input(description="Text prompt for segmentation", default="person"),
        threshold: float = Input(
            description="Confidence threshold for object detection",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        save_overlay: bool = Input(
            description="If True, includes the overlay image in the ZIP file",
            default=False
        ),
        mask_only: bool = Input(
            description="If True, returns a black-and-white mask image instead of an overlay on the original image",
            default=False
        ),
        return_zip: bool = Input(
            description="If True, returns a ZIP file containing individual masks as PNGs instead of a single image",
            default=False
        ),
        mask_opacity: float = Input(
            description="Opacity of the mask overlay (0.0 to 1.0)",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        mask_color: str = Input(
            description="Color of the mask overlay. Options: 'green', 'red', 'blue', 'yellow', 'cyan', 'magenta'",
            default="green"
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # 1. Load image
        print(f"Processing image: {image}", flush=True)
        pil_image = Image.open(str(image)).convert("RGB")
        
        # 2. Prepare inputs
        print(f"Adding text prompt: '{prompt}'", flush=True)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        
        print(f"Input keys: {list(inputs.keys())}", flush=True)
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                print(f"  {key}: shape={inputs[key].shape}, dtype={inputs[key].dtype}", flush=True)
        
        # Cast float tensors to the correct dtype
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
                inputs[key] = inputs[key].to(dtype=self.dtype)
        
        # 3. Inference
        print(f"Running inference on {self.device} with {self.dtype}...", flush=True)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        print("Inference complete!", flush=True)
            
        # 4. Post-process results
        target_sizes = inputs.get("original_sizes").tolist()
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=target_sizes
        )[0]
        
        masks = results['masks'] # Binary masks
        print(f"Found {len(masks)} objects")
        
        # 5. Generate output
        output_dir = Path("/tmp/output_masks")
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        if return_zip:
            # Save individual masks
            for i, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(os.path.join(output_dir, f"mask_{i:03d}.png"))
            
            # Also save the overlay image for reference
            if save_overlay:
                overlay_path = os.path.join(output_dir, "overlay.png")
                self._save_image(pil_image, masks, overlay_path, mask_opacity=mask_opacity, mask_color=mask_color, mask_only=mask_only)

            # Create Zip
            import shutil
            output_zip_path = Path("/tmp/output.zip")
            shutil.make_archive("/tmp/output", 'zip', output_dir)
            return output_zip_path
            
        else:
            output_path = Path("/tmp/output.png")
            self._save_image(pil_image, masks, str(output_path), mask_opacity=mask_opacity, mask_color=mask_color, mask_only=mask_only)
            return output_path

    def _save_image(self, image, masks, output_path, mask_opacity=0.5, mask_color="green", mask_only=False):
        print(f"Saving output image to {output_path}...")
        width, height = image.size
        
        # Define colors
        colors = {
            "green": [0, 255, 0],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255]
        }
        color_rgb = np.array(colors.get(mask_color.lower(), [0, 255, 0]), dtype=np.uint8)
        
        image_np = np.array(image)
        
        if mask_only:
            output_image = np.zeros_like(image_np)
        else:
            output_image = image_np.copy()
            
        if masks is not None and len(masks) > 0:
            combined_mask = np.zeros((height, width), dtype=bool)
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Handle dimensions
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.ndim > 2:
                    mask = mask.squeeze()
                    
                if mask.shape != (height, width):
                    # Resize mask if needed
                    mask_img = Image.fromarray(mask.astype(np.uint8))
                    mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
                    mask = np.array(mask_img)
                
                combined_mask = np.logical_or(combined_mask, mask > 0.0)
            
            overlay_indices = combined_mask
            
            if mask_only:
                output_image[overlay_indices] = [255, 255, 255]
            else:
                # Color overlay
                output_image[overlay_indices] = (output_image[overlay_indices].astype(float) * (1 - mask_opacity) + color_rgb.astype(float) * mask_opacity).astype(np.uint8)

        Image.fromarray(output_image).save(output_path)
        print("Image saved.")
