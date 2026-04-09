import argparse
import sys
import os
import numpy as np
from pathlib import Path

import cv2
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
REPO_ROOT = SRC_DIR.parent
for path in (CURRENT_DIR, SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from common.config import (
    build_frame_filename,
    collect_image_paths,
    ensure_phase_output_dirs,
    get_phase_output_dir,
    load_yaml_config,
    resolve_sequence_spec,
)
from common.mask_utils import postprocess_mask
from common.visualization import (
    ensure_visualization_dirs,
    render_before_after_comparison,
    render_instance_mask_overlay,
    render_mask_overlay,
    save_visualization_frame,
)
try:
    from part2_sota.mask_sam2 import SAM2MaskGenerator
    from part2_sota.inpaint_pro_painter import ProPainterInpainter
except ModuleNotFoundError as exc:
    if exc.name != "part2_sota":
        raise
    from mask_sam2 import SAM2MaskGenerator
    from inpaint_pro_painter import ProPainterInpainter


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


def normalize_sam2_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 2:
        return None
    return (mask > 0).astype(np.uint8) * 255


def save_object_masks(video_segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, filename))
    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            mask_path = os.path.join(output_dir, f"frame_{frame_idx:04d}_obj{obj_id}.png")
            mask_img = normalize_sam2_mask(mask)
            if mask_img is not None:
                Image.fromarray(mask_img).save(mask_path)
            else:
                print(f"Warning: Unexpected mask shape {mask.shape} for frame {frame_idx}, obj {obj_id}")


def build_processed_masks(video_segments, frame_files, postprocess_cfg):
    processed_masks = []
    object_masks_per_frame = []
    max_history = max(int(postprocess_cfg.get("temporal_window", 1)) - 1, 0)
    mask_history = []

    for idx, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame for postprocessing: {frame_path}")

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        frame_object_masks = []
        for obj_id, mask in sorted(video_segments.get(idx, {}).items()):
            normalized_mask = normalize_sam2_mask(mask)
            if normalized_mask is None:
                print(f"Warning: Skipping malformed mask for frame {idx}, obj {obj_id}")
                continue
            if normalized_mask.shape != combined_mask.shape:
                normalized_mask = cv2.resize(
                    normalized_mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            frame_object_masks.append((obj_id, normalized_mask))
            combined_mask = np.maximum(combined_mask, normalized_mask)

        processed_mask = postprocess_mask(
            combined_mask,
            postprocess_cfg=postprocess_cfg,
            previous_masks=mask_history,
        )
        if processed_mask is None:
            processed_mask = combined_mask

        processed_masks.append(processed_mask)
        object_masks_per_frame.append(frame_object_masks)
        if max_history > 0:
            mask_history.append(processed_mask)
            mask_history = mask_history[-max_history:]

    return object_masks_per_frame, processed_masks


def save_combined_masks(processed_masks, frame_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, filename))
    for idx, frame_path in enumerate(frame_files):
        combined_mask = processed_masks[idx]
        output_name = f"{Path(frame_path).stem}.png"
        Image.fromarray(combined_mask).save(os.path.join(output_dir, output_name))


def export_visualizations(
    frame_files,
    object_masks_per_frame,
    processed_masks,
    restored_frames,
    visualization_root,
    sequence_name,
    common_cfg,
    visualization_cfg,
):
    if not visualization_cfg.get("enabled", False):
        return

    visualization_dirs = ensure_visualization_dirs(visualization_root, sequence_name)
    overlay_alpha = visualization_cfg.get("overlay_alpha", 0.35)
    mask_overlay_frames = []
    original_frames = []

    for index, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame for visualization: {frame_path}")
        original_frames.append(frame)

        object_masks = [mask for _, mask in object_masks_per_frame[index]]
        object_labels = [f"obj={obj_id}" for obj_id, _ in object_masks_per_frame[index]]
        if visualization_cfg.get("save_object_overlays", True):
            object_overlay_frame = render_instance_mask_overlay(
                frame,
                object_masks,
                instance_labels=object_labels,
                overlay_alpha=overlay_alpha,
                title="SAM2 objects",
            )
            object_output_path = os.path.join(
                visualization_dirs["motion_scores"],
                build_frame_filename(common_cfg, index, artifact="frame"),
            )
            save_visualization_frame(object_overlay_frame, object_output_path)

        mask_overlay_frame = render_mask_overlay(
            frame,
            processed_masks[index],
            overlay_alpha=overlay_alpha,
        )
        mask_overlay_frames.append(mask_overlay_frame)
        if visualization_cfg.get("save_mask_overlays", True):
            mask_output_path = os.path.join(
                visualization_dirs["mask_overlays"],
                build_frame_filename(common_cfg, index, artifact="frame"),
            )
            save_visualization_frame(mask_overlay_frame, mask_output_path)

    if visualization_cfg.get("save_comparisons", True):
        for index, restored_frame in enumerate(restored_frames):
            restored_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)
            comparison_frame = render_before_after_comparison(
                original_frames[index],
                mask_overlay_frames[index],
                restored_bgr,
            )
            comparison_output_path = os.path.join(
                visualization_dirs["comparisons"],
                build_frame_filename(common_cfg, index, artifact="frame"),
            )
            save_visualization_frame(comparison_frame, comparison_output_path)

    print(f"Saved visualizations under {visualization_dirs['base']}")


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
    pipeline_cfg = phase_cfg.get("pipeline", {})
    postprocess_cfg = pipeline_cfg.get("postprocess", {})
    visualization_cfg = pipeline_cfg.get("visualization", {})

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

    masks_dir = get_phase_output_dir(phase_cfg, "masks_dir") or "results/masks/part2"
    sequence_masks_dir = os.path.join(masks_dir, sequence_spec["output_name"])
    object_masks_dir = os.path.join(sequence_masks_dir, "objects")
    combined_masks_dir = os.path.join(sequence_masks_dir, "combined")

    os.makedirs(object_masks_dir, exist_ok=True)
    save_object_masks(video_segments, object_masks_dir)

    frame_files = collect_image_paths(input_frames_dir, common_cfg) or list_image_files(input_frames_dir)
    object_masks_per_frame, processed_masks = build_processed_masks(
                video_segments,
                frame_files,
                postprocess_cfg,
        )
    save_combined_masks(processed_masks, frame_files, combined_masks_dir)

    output_video_dir = get_phase_output_dir(phase_cfg, "videos_dir") or "results/videos/part2"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{sequence_spec['output_name']}_inpainted.mp4")

    print("Inpainting with ProPainter...")
    restored_frames = inpainter.inpaint(input_frames_dir, combined_masks_dir, output_video_path)

    visualization_root = get_phase_output_dir(phase_cfg, "visualizations_dir") or "results/visualizations/part2"
    export_visualizations(
        frame_files,
        object_masks_per_frame,
        processed_masks,
        restored_frames,
        visualization_root,
        sequence_spec["output_name"],
        common_cfg,
        visualization_cfg,
    )

    print("Part 2 pipeline finished.")
    print(f"Saved object masks to: {object_masks_dir}")
    print(f"Saved combined masks to: {combined_masks_dir}")
    print(f"Saved inpainted video to: {output_video_path}")


if __name__ == "__main__":
    main()