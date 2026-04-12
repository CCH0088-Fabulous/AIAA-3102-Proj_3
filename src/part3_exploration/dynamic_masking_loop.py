import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
REPO_ROOT = SRC_DIR.parent
for path in (CURRENT_DIR, SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from common.config import (
    collect_image_paths,
    ensure_phase_output_dirs,
    get_phase_output_dir,
    load_yaml_config,
    resolve_sequence_spec,
    build_frame_filename,
)

from part2_sota.mask_sam2 import SAM2MaskGenerator
from part3_exploration.sam3_upgrade import SAM3UpgradeRefiner

def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: Dynamic Closed-Loop Masking (SAM2 + SAM3)")
    parser.add_argument("--sequence", default="bmx-trees", help="Sequence key")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument("--part2-config", default="configs/part2_sota.yaml")
    parser.add_argument("--part3-config", default="configs/part3_exploration.yaml")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames for quick test")
    return parser.parse_args()


def load_prompts_from_config(config_data, sequence_name):
    raw_prompts = config_data.get("prompts", {}).get(sequence_name, [])
    prompts = []
    for p in raw_prompts:
        prompt_dict = {
            "frame_idx": p["frame_idx"],
            "obj_id": p.get("obj_id", 1)
        }
        if "box" in p:
            prompt_dict["box"] = np.array(p["box"], dtype=np.float32)
        if "points" in p and "labels" in p:
            prompt_dict["points"] = np.array(p["points"], dtype=np.float32)
            prompt_dict["labels"] = np.array(p["labels"], dtype=np.int32)
        prompts.append(prompt_dict)
    return prompts


def main():
    args = parse_args()
    
    common_cfg = load_yaml_config(args.common_config)
    part2_cfg = load_yaml_config(args.part2_config)
    part3_cfg = load_yaml_config(args.part3_config)
    
    sequence_spec = resolve_sequence_spec(args.sequence, common_cfg)
    input_frames_dir = sequence_spec["frames_dir"]
    frame_files = collect_image_paths(input_frames_dir, common_cfg)
    
    if args.max_frames is not None:
        frame_files = frame_files[:args.max_frames]
        
    print(f"--- Starting Dynamic Closed-Loop Masking System ---")
    print(f"Sequence: {sequence_spec['output_name']}, Frames: {len(frame_files)}")
    
    # 1. Initialize SAM2 tracker
    print("\n[Step 1] Initializing SAM2 Tracker...")
    sam2_weights = part2_cfg.get("models", {}).get("segmentation", {}).get("weights_dir", "models/sam2")
    sam2_tracker = SAM2MaskGenerator(weights_dir=sam2_weights)
    
    prompts = load_prompts_from_config(part2_cfg, args.sequence)
    if not prompts:
        print(f"Warning: No prompts found for sequence {args.sequence}")
        
    print(f"[Step 1] Running SAM2 Tracking...")
    video_segments = sam2_tracker.generate(input_frames_dir, prompts=prompts)
    
    # Convert video_segments to the format expected by SAM3UpgradeRefiner (coarse_masks)
    # The SAM2 logic outputs: dict: {frame_idx: {obj_id: mask_array}}
    # We will build a unified coarse mask for each frame
    coarse_masks = []
    for frame_idx in range(len(frame_files)):
        frame = cv2.imread(frame_files[frame_idx])
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                # mask is binary array or probability mapping, normalize it locally
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8) * 255)
        coarse_masks.append(combined_mask)
        
    print(f"\n[Step 2] Initializing SAM3 Refiner...")
    seg3_cfg = part3_cfg.get("models", {}).get("segmentation_upgrade", {})
    refinement_cfg = part3_cfg.get("pipeline", {}).get("refinement", {})
    
    # Instantiate SAM3 Refiner
    sam3_refiner = SAM3UpgradeRefiner(
        weights_dir=seg3_cfg.get("weights_dir", "models/sam3"),
        checkpoint_dir=seg3_cfg.get("checkpoint_dir", "models/sam3/checkpoints"),
        checkpoint_name=seg3_cfg.get("checkpoint_name"),
        repo_id=seg3_cfg.get("repo_id"),
        auto_download=seg3_cfg.get("auto_download", True),
        confidence_threshold=refinement_cfg.get("confidence_threshold", 0.35),
        box_expand_ratio=refinement_cfg.get("box_expand_ratio", 0.08),
        # You can add the rest of refinement bounds if needed
    )
    
    print(f"[Step 2] Refining coarse SAM2 masks with SAM3...")
    candidate_masks_per_frame, raw_refined_masks = sam3_refiner.refine(frame_files, coarse_masks)
    
    output_dir = os.path.join("results", "masks", "dynamic_loop", sequence_spec["output_name"])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[Finished] Saving refined masks to: {output_dir}")
    for idx, (frame_path, processed_mask) in enumerate(zip(frame_files, raw_refined_masks)):
        output_name = f"frame_{idx:04d}.png"
        Image.fromarray(processed_mask).save(os.path.join(output_dir, output_name))
        
    print("Done! Dynamic closed-loop masking run successfully.")

if __name__ == "__main__":
    main()