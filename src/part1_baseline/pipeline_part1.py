import argparse
from collections import deque
import os
import sys
from pathlib import Path

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.config import (
    build_frame_filename,
    collect_image_paths,
    ensure_phase_output_dirs,
    get_phase_output_dir,
    load_yaml_config,
    resolve_sequence_spec,
)
from common.mask_utils import postprocess_mask
from dynamic_judgment import DynamicObjectJudge
from mask_extraction_yolo import MaskExtractorYOLO
from utils import build_mask_output_path, merge_instance_masks, save_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 1 baseline mask extraction.")
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence key from configs/common.yaml or a direct folder containing frames.",
    )
    parser.add_argument(
        "--folder",
        dest="sequence_alias",
        default=None,
        help=(
            "Backward-compatible alias for --sequence. Supports a dataset key from configs/common.yaml, "
            "a direct folder path, or a folder name under data/raw."
        ),
    )
    parser.add_argument(
        "--common-config",
        default="configs/common.yaml",
        help="Path to the shared project configuration YAML.",
    )
    parser.add_argument(
        "--phase-config",
        default="configs/part1_baseline.yaml",
        help="Path to the Part 1 phase configuration YAML.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on how many frames to process, useful for debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    common_cfg = load_yaml_config(args.common_config)
    baseline_cfg = load_yaml_config(args.phase_config)
    ensure_phase_output_dirs(baseline_cfg)

    default_sequence = baseline_cfg.get("pipeline", {}).get("input", {}).get(
        "sequence_key",
        common_cfg.get("project", {}).get("default_sequence", "bmx-trees"),
    )
    sequence_arg = args.sequence or args.sequence_alias or default_sequence
    sequence_spec = resolve_sequence_spec(sequence_arg, common_cfg)

    datasets_dir = sequence_spec["frames_dir"]
    output_mask_dir = get_phase_output_dir(baseline_cfg, "masks_dir")
    sequence_name = sequence_spec["output_name"]
    mask_dir = build_mask_output_path(output_mask_dir, sequence_name)

    segmentation_cfg = baseline_cfg.get("models", {}).get("segmentation", {})
    pipeline_cfg = baseline_cfg.get("pipeline", {})
    dynamic_cfg = pipeline_cfg.get("dynamic_filter", {})
    postprocess_cfg = pipeline_cfg.get("postprocess", {})
    target_classes = baseline_cfg.get("pipeline", {}).get("mask_extraction", {}).get(
        "target_classes",
        [],
    )

    extractor = MaskExtractorYOLO(
        model_path=segmentation_cfg["weights"],
        target_classes=target_classes,
    )

    image_paths = collect_image_paths(datasets_dir, common_cfg)
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    if not image_paths:
        print(f"No images found in {datasets_dir}!")
        return

    dynamic_judge = DynamicObjectJudge(
        motion_threshold=dynamic_cfg.get("motion_threshold", 1.5),
        min_tracked_points=dynamic_cfg.get("min_tracked_points", 8),
        aggregation=dynamic_cfg.get("aggregation", "median"),
        keep_if_undetermined=dynamic_cfg.get("keep_if_undetermined", True),
        feature_params=dynamic_cfg.get("feature_params", {}),
        lk_params=dynamic_cfg.get("lk_params", {}),
    )
    mask_history = deque(maxlen=max(int(postprocess_cfg.get("temporal_window", 1)) - 1, 0))
    previous_frame = None

    print(f"Using dataset folder: {datasets_dir}")
    print(f"Canonical sequence name: {sequence_name}")
    print(f"Found {len(image_paths)} frames. Starting processing...")

    for i, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        masks, bboxes, class_ids = extractor.extract(frame)
        if dynamic_cfg.get("enabled", False):
            dynamic_masks, motion_summaries = dynamic_judge.filter_dynamic_instances(
                previous_frame,
                frame,
                masks,
            )
        else:
            dynamic_masks = masks
            motion_summaries = []

        merged_mask = merge_instance_masks(dynamic_masks, frame.shape)
        merged_mask = postprocess_mask(
            merged_mask,
            postprocess_cfg=postprocess_cfg,
            previous_masks=list(mask_history),
        )

        mask_path = os.path.join(mask_dir, build_frame_filename(common_cfg, i, artifact="mask"))
        save_mask(merged_mask, mask_path)

        kept_instances = len(dynamic_masks)
        total_instances = len(masks)
        tracked_instances = sum(1 for summary in motion_summaries if summary.get("valid"))
        print(
            f"Processed frame {i+1}/{len(image_paths)} | "
            f"instances kept: {kept_instances}/{total_instances} | "
            f"motion-validated: {tracked_instances}"
        )

        if merged_mask is not None:
            mask_history.append(merged_mask)
        previous_frame = frame

if __name__ == '__main__':
    main()
