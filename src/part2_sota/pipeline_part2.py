import argparse
import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.config import (
    ensure_phase_output_dirs,
    load_yaml_config,
    resolve_sequence_spec,
)
try:
    from mask_sam2 import SAM2MaskGenerator
    from inpaint_pro_painter import ProPainterInpainter
except ModuleNotFoundError:
    from src.part2_sota.mask_sam2 import SAM2MaskGenerator
    from src.part2_sota.inpaint_pro_painter import ProPainterInpainter


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2 SOTA pipeline.")
    parser.add_argument("--sequence", default=None, help="Sequence key or direct frame folder.")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument("--phase-config", default="configs/part2_sota.yaml")
    parser.add_argument(
        "--prompts",
        nargs='+',
        help=(
            "Prompts for SAM2. Two formats are supported: "
            "frame_idx obj_id x y label  OR  frame_idx obj_id box x0 y0 x1 y1"
        ),
    )
    return parser.parse_args()


def parse_prompts(prompts_args):
    """Parse SAM2 prompt arguments from the CLI."""
    prompts = []
    if not prompts_args:
        return prompts

    idx = 0
    while idx < len(prompts_args):
        frame_idx = int(prompts_args[idx]); idx += 1
        obj_id = int(prompts_args[idx]); idx += 1
        if idx < len(prompts_args) and prompts_args[idx].lower() == "box":
            idx += 1
            if idx + 4 > len(prompts_args):
                raise ValueError("Box prompt requires 4 values: x0 y0 x1 y1")
            x0 = float(prompts_args[idx]); y0 = float(prompts_args[idx + 1])
            x1 = float(prompts_args[idx + 2]); y1 = float(prompts_args[idx + 3])
            idx += 4
            prompts.append(
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "box": np.array([x0, y0, x1, y1], dtype=np.float32),
                }
            )
        else:
            if idx + 3 >= len(prompts_args):
                raise ValueError("Each point prompt must contain 5 values: frame_idx obj_id x y label")
            x = float(prompts_args[idx]); y = float(prompts_args[idx + 1])
            label = int(prompts_args[idx + 2])
            idx += 3
            prompts.append(
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": np.array([[x, y]], dtype=np.float32),
                    "labels": np.array([label], dtype=np.int32),
                }
            )
    return prompts


def list_image_files(image_dir):
    supported = {".jpg", ".jpeg", ".png"}
    return sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if os.path.splitext(fname)[1].lower() in supported
        ]
    )


def save_object_masks(video_segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            mask_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_obj{obj_id}.png")
            # Handle SAM2 output format: (1, H, W) -> (H, W)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # Remove batch dimension
            if mask.ndim == 2:
                mask_img = (mask.astype(np.uint8) * 255)
                Image.fromarray(mask_img).save(mask_path)
            else:
                print(f"Warning: Unexpected mask shape {mask.shape} for frame {frame_idx}, obj {obj_id}")


def save_combined_masks(video_segments, frame_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    zero_mask = None
    for idx, frame_path in enumerate(frame_files):
        if zero_mask is None:
            if idx in video_segments and video_segments[idx]:
                first_mask = next(iter(video_segments[idx].values()))
                if first_mask.ndim == 3 and first_mask.shape[0] == 1:
                    first_mask = first_mask.squeeze(0)
                zero_mask = np.zeros_like(first_mask, dtype=np.uint8)
            else:
                zero_mask = np.zeros((240, 432), dtype=np.uint8)  # Based on debug output
        combined_mask = np.zeros_like(zero_mask, dtype=np.uint8)
        if idx in video_segments:
            for mask in video_segments[idx].values():
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                combined_mask = np.maximum(combined_mask, (mask.astype(np.uint8) * 255))
        output_name = os.path.basename(frame_path)
        Image.fromarray(combined_mask).save(os.path.join(output_dir, output_name))


def main():
    args = parse_args()
    common_cfg = load_yaml_config(args.common_config)
    phase_cfg = load_yaml_config(args.phase_config)
    ensure_phase_output_dirs(phase_cfg)

    default_sequence = phase_cfg.get("pipeline", {}).get("input", {}).get(
        "sequence_key",
        common_cfg.get("project", {}).get("default_sequence", "bmx-trees"),
    )
    sequence_spec = resolve_sequence_spec(args.sequence or default_sequence, common_cfg)

    sam2_weights = phase_cfg.get("models", {}).get("segmentation", {}).get("weights_dir", "models/sam2")
    propainter_weights = phase_cfg.get("models", {}).get("inpainting", {}).get("weights_dir", "models/ProPainter")

    print(f"Sequence: {sequence_spec['output_name']}")
    print(f"Frames: {sequence_spec['frames_dir']}")
    print(f"SAM2 weights: {sam2_weights}")
    print(f"ProPainter weights: {propainter_weights}")

    mask_generator = SAM2MaskGenerator(sam2_weights)
    inpainter = ProPainterInpainter(propainter_weights)

    input_frames_dir = sequence_spec["frames_dir"]
    if not os.path.isdir(input_frames_dir):
        raise FileNotFoundError(f"Input frames directory not found: {input_frames_dir}")

    prompts = parse_prompts(args.prompts)
    if not prompts:
        # Try to load prompts from config file
        config_prompts = phase_cfg.get("prompts", {}).get(sequence_spec["output_name"], [])
        if config_prompts:
            prompts = []
            for p in config_prompts:
                prompt = {
                    "frame_idx": p["frame_idx"],
                    "obj_id": p["obj_id"],
                }
                if "box" in p:
                    prompt["box"] = np.array(p["box"], dtype=np.float32)
                elif "points" in p and "labels" in p:
                    prompt["points"] = np.array(p["points"], dtype=np.float32)
                    prompt["labels"] = np.array(p["labels"], dtype=np.int32)
                prompts.append(prompt)
            print(f"Using configured prompts for {sequence_spec['output_name']} from config file.")
        else:
            # Fallback to default prompts based on sequence name
            frame_files = list_image_files(input_frames_dir)
            if not frame_files:
                raise FileNotFoundError(f"No image frames found in {input_frames_dir}")
            first_image = Image.open(frame_files[0])
            width, height = first_image.size
            if "bmx-trees" in sequence_spec["output_name"]:
                # Use a stronger default box prompt for the BMX sequence so SAM2 covers the full person+bike.
                prompts = [
                    {
                        "frame_idx": 0,
                        "obj_id": 1,
                        "box": np.array([100.0, 40.0, 380.0, 220.0], dtype=np.float32),
                    }
                ]
                print("Using default bmx-trees box prompt for SAM2 mask generation.")
            elif "tennis" in sequence_spec["output_name"]:
                # Tennis scene: person on left, racket and ball on right, shadows on ground
                prompts = [
                    {
                        "frame_idx": 0,
                        "obj_id": 1,  # Person
                        "box": np.array([40.0, 40.0, 180.0, 215.0], dtype=np.float32),
                        "points": np.array([[90.0, 120.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    },
                    {
                        "frame_idx": 0,
                        "obj_id": 2,  # Tennis racket
                        "box": np.array([160.0, 70.0, 240.0, 160.0], dtype=np.float32),
                        "points": np.array([[205.0, 120.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    },
                    {
                        "frame_idx": 0,
                        "obj_id": 3,  # Tennis ball
                        "box": np.array([295.0, 145.0, 325.0, 175.0], dtype=np.float32),
                        "points": np.array([[310.0, 160.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    },
                    {
                        "frame_idx": 0,
                        "obj_id": 4,  # Person shadow
                        "box": np.array([20.0, 190.0, 150.0, 240.0], dtype=np.float32),
                        "points": np.array([[70.0, 220.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    },
                    {
                        "frame_idx": 0,
                        "obj_id": 5,  # Racket/ball shadow
                        "box": np.array([220.0, 205.0, 325.0, 240.0], dtype=np.float32),
                        "points": np.array([[255.0, 225.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    }
                ]
                print("Using default tennis box+point prompts for SAM2 mask generation (person, racket, ball, shadows).")
            else:
                prompts = [
                    {
                        "frame_idx": 0,
                        "obj_id": 1,
                        "points": np.array([[width / 2.0, height / 2.0]], dtype=np.float32),
                        "labels": np.array([1], dtype=np.int32),
                    }
                ]
                print(f"Using default center point prompt for {sequence_spec['output_name']} (no specific config found).")

    print("Generating SAM2 masks...")
    video_segments = mask_generator.generate(input_frames_dir, prompts)

    masks_dir = phase_cfg.get("phase", {}).get("outputs", {}).get("masks_dir", "results/masks/part2")
    sequence_masks_dir = os.path.join(masks_dir, sequence_spec["output_name"])
    object_masks_dir = os.path.join(sequence_masks_dir, "objects")
    combined_masks_dir = os.path.join(sequence_masks_dir, "combined")

    os.makedirs(object_masks_dir, exist_ok=True)
    save_object_masks(video_segments, object_masks_dir)

    frame_files = list_image_files(input_frames_dir)
    save_combined_masks(video_segments, frame_files, combined_masks_dir)

    output_video_dir = phase_cfg.get("phase", {}).get("outputs", {}).get("videos_dir", "results/videos/part2")
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{sequence_spec['output_name']}_inpainted.mp4")

    print("Inpainting with ProPainter...")
    inpainter.inpaint(input_frames_dir, combined_masks_dir, output_video_path)

    print("Part 2 pipeline finished.")
    print(f"Saved object masks to: {object_masks_dir}")
    print(f"Saved combined masks to: {combined_masks_dir}")
    print(f"Saved inpainted video to: {output_video_path}")


if __name__ == "__main__":
    main()