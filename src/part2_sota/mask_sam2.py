import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Fix OpenCV import path for conda-installed cv2
def _ensure_cv2_path():
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cv2_dir = Path(sys.prefix) / "lib" / py_ver / "site-packages" / "cv2" / f"python-{sys.version_info.major}.{sys.version_info.minor}"
    if cv2_dir.exists() and str(cv2_dir) not in sys.path:
        sys.path.insert(0, str(cv2_dir))

_ensure_cv2_path()
import cv2
import torch

# Add SAM2 to path
SAM2_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "sam2"
if str(SAM2_DIR) not in sys.path:
    sys.path.append(str(SAM2_DIR))

def _initialize_sam2_hydra():
    config_dir = Path(SAM2_DIR) / "sam2" / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(config_dir), job_name="sam2_init", version_base=None)

from sam2.build_sam import build_sam2_video_predictor


class SAM2MaskGenerator:
    """SAM2-based mask generator for video object segmentation."""

    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)
        self.predictor = None
        self._load_model()

    def _load_model(self):
        """Load SAM2 model and predictor."""
        checkpoint = None
        for candidate in [
            "sam2.1_hiera_large.pt",
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_small.pt",
            "sam2.1_hiera_tiny.pt",
        ]:
            candidate_path = self.weights_dir / "checkpoints" / candidate
            if candidate_path.exists():
                checkpoint = str(candidate_path)
                break

        if checkpoint is None:
            raise FileNotFoundError(
                f"No SAM2 checkpoint found in {self.weights_dir / 'checkpoints'}"
            )

        config_map = {
            "sam2.1_hiera_large.pt": "sam2.1/sam2.1_hiera_l",
            "sam2.1_hiera_base_plus.pt": "sam2.1/sam2.1_hiera_b+",
            "sam2.1_hiera_small.pt": "sam2.1/sam2.1_hiera_s",
            "sam2.1_hiera_tiny.pt": "sam2.1/sam2.1_hiera_t",
        }
        config_name = config_map[Path(checkpoint).name]

        _initialize_sam2_hydra()
        self.predictor = build_sam2_video_predictor(config_name, checkpoint, vos_optimized=False)

    def generate(self, frames_dir, prompts=None):
        """
        Generate masks for video frames.

        Args:
            frames_dir (str): Path to directory containing video frames (JPEG/PNG).
            prompts (list): List of prompts for objects. Each prompt is a dict with:
                - 'frame_idx': Frame index to prompt on
                - 'obj_id': Object ID
                - 'points': np.array of shape (N, 2) for click points
                - 'labels': np.array of shape (N,) for point labels (1=positive, 0=negative)
                - 'box': np.array of shape (4,) for bounding box [x_min, y_min, x_max, y_max]

        Returns:
            dict: {frame_idx: {obj_id: mask_array}} where mask_array is binary numpy array
        """
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Check if we need to convert PNG to JPG for SAM2
        png_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.PNG'))]
        use_temp_dir = len(png_files) > 0

        if use_temp_dir:
            import tempfile
            import shutil
            from PIL import Image

            temp_dir = tempfile.mkdtemp()
            try:
                # Copy/convert all frames to JPG in temp directory
                for filename in sorted(os.listdir(frames_dir)):
                    src_path = os.path.join(frames_dir, filename)
                    if os.path.isfile(src_path):
                        # Convert to JPG regardless of original format
                        img = Image.open(src_path).convert('RGB')
                        base_name = os.path.splitext(filename)[0]
                        dst_path = os.path.join(temp_dir, f"{base_name}.jpg")
                        img.save(dst_path, 'JPEG')
                
                video_dir = temp_dir
            except Exception as e:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise e
        else:
            video_dir = frames_dir

        try:
            # Initialize inference state
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = self.predictor.init_state(video_path=video_dir)

                # Add prompts if provided
                if prompts:
                    for prompt in prompts:
                        frame_idx = prompt['frame_idx']
                        obj_id = prompt['obj_id']
                        box = prompt.get('box', None)
                        points = prompt.get('points', None)
                        labels = prompt.get('labels', None)

                        if points is not None and labels is not None:
                            self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                points=points,
                                labels=labels,
                                box=box,
                            )
                        elif box is not None:
                            self.predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                box=box,
                            )
                        elif points is not None or labels is not None:
                            raise ValueError(
                                f"Prompt for obj_id={obj_id} must include both points and labels or a box."
                            )

                # Propagate to get masks for all frames
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

        finally:
            if use_temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return video_segments