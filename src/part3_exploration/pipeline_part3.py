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
    build_frame_filename,
    build_video_filename,
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
    from part2_sota.inpaint_pro_painter import ProPainterInpainter
    from part3_exploration.sam3_upgrade import SAM3UpgradeRefiner
except ModuleNotFoundError as exc:
    if exc.name not in {"part2_sota", "part3_exploration"}:
        raise
    from part2_sota.inpaint_pro_painter import ProPainterInpainter
    from sam3_upgrade import SAM3UpgradeRefiner


def parse_args():
    parser = argparse.ArgumentParser(description="Part 3 exploration pipeline with SAM3 refinement.")
    parser.add_argument("--sequence", default=None, help="Sequence key or direct frame folder.")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument("--phase-config", default="configs/part3_exploration.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def read_image(path, mode="RGB"):
    image = Image.open(path)
    if mode == "L":
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    return np.array(image)


def resolve_baseline_mask_dir(common_cfg, phase_cfg, sequence_name):
    configured_dir = phase_cfg.get("pipeline", {}).get("baseline_masks_dir_path")
    if configured_dir:
        return configured_dir

    masks_root = common_cfg.get("paths", {}).get("masks_root", "results/masks")
    baseline_phase = phase_cfg.get("pipeline", {}).get("baseline_phase", "part2")
    baseline_masks_subdir = phase_cfg.get("pipeline", {}).get("baseline_masks_subdir", "combined")
    return os.path.join(masks_root, baseline_phase, sequence_name, baseline_masks_subdir)


def resolve_baseline_object_mask_dir(common_cfg, phase_cfg, sequence_name):
    configured_dir = phase_cfg.get("pipeline", {}).get("baseline_objects_dir_path")
    if configured_dir:
        return configured_dir

    masks_root = common_cfg.get("paths", {}).get("masks_root", "results/masks")
    baseline_phase = phase_cfg.get("pipeline", {}).get("baseline_phase", "part2")
    baseline_objects_subdir = phase_cfg.get("pipeline", {}).get("baseline_objects_subdir", "objects")
    return os.path.join(masks_root, baseline_phase, sequence_name, baseline_objects_subdir)


def load_coarse_masks(frame_files, masks_dir, common_cfg):
    mask_paths = collect_image_paths(masks_dir, common_cfg)
    mask_by_stem = {Path(mask_path).stem: read_image(mask_path, mode="L") for mask_path in mask_paths}

    coarse_masks = []
    for frame_path in frame_files:
        frame = read_image(frame_path, mode="RGB")
        fallback = np.zeros(frame.shape[:2], dtype=np.uint8)
        coarse_masks.append(mask_by_stem.get(Path(frame_path).stem, fallback))
    return coarse_masks


def load_object_masks(frame_files, objects_dir):
    object_masks_per_frame = []
    for frame_index, frame_path in enumerate(frame_files):
        frame = read_image(frame_path, mode="RGB")
        frame_shape = frame.shape[:2]
        masks = []
        if os.path.isdir(objects_dir):
            prefix = f"frame_{frame_index:04d}_obj"
            object_paths = sorted(
                os.path.join(objects_dir, filename)
                for filename in os.listdir(objects_dir)
                if filename.startswith(prefix) and filename.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            for object_path in object_paths:
                object_mask = read_image(object_path, mode="L")
                if object_mask.shape != frame_shape:
                    object_mask = cv2.resize(
                        object_mask,
                        (frame_shape[1], frame_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                object_id_text = Path(object_path).stem.split("_obj")[-1]
                try:
                    object_id = int(object_id_text)
                except ValueError:
                    object_id = len(masks) + 1
                masks.append((object_id, object_mask))
        object_masks_per_frame.append(masks)
    return object_masks_per_frame


def build_union_masks(object_masks_per_frame, frame_files):
    union_masks = []
    for frame_index, frame_path in enumerate(frame_files):
        frame = read_image(frame_path, mode="RGB")
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for _, object_mask in object_masks_per_frame[frame_index]:
            combined_mask = np.maximum(combined_mask, (object_mask > 0).astype(np.uint8) * 255)
        union_masks.append(combined_mask)
    return union_masks


def resolve_baseline_mask_source(phase_cfg, object_masks_per_frame):
    source = phase_cfg.get("pipeline", {}).get("baseline_mask_source", "combined")
    if source != "auto":
        return source

    max_objects_per_frame = max((len(frame_masks) for frame_masks in object_masks_per_frame), default=0)
    if max_objects_per_frame <= 1 and max_objects_per_frame > 0:
        return "objects_union"
    return "combined"


def save_candidate_masks(candidate_masks_per_frame, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, filename))

    for frame_index, candidates in enumerate(candidate_masks_per_frame):
        for candidate_index, candidate in enumerate(candidates, start=1):
            mask_path = os.path.join(output_dir, f"frame_{frame_index:04d}_obj{candidate_index}.png")
            Image.fromarray(candidate["mask"]).save(mask_path)


def save_combined_masks(processed_masks, frame_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(output_dir, filename))

    for frame_path, processed_mask in zip(frame_files, processed_masks):
        output_name = f"{Path(frame_path).stem}.png"
        Image.fromarray(processed_mask).save(os.path.join(output_dir, output_name))


def postprocess_masks(raw_masks, postprocess_cfg):
    processed_masks = []
    max_history = max(int(postprocess_cfg.get("temporal_window", 1)) - 1, 0)
    mask_history = []

    for raw_mask in raw_masks:
        processed_mask = postprocess_mask(raw_mask, postprocess_cfg=postprocess_cfg, previous_masks=mask_history)
        if processed_mask is None:
            processed_mask = raw_mask
        processed_masks.append(processed_mask)
        if max_history > 0:
            mask_history.append(processed_mask)
            mask_history = mask_history[-max_history:]

    return processed_masks


def normalize_masks(raw_masks):
    return [((np.asarray(raw_mask) > 0).astype(np.uint8) * 255) for raw_mask in raw_masks]


def export_visualizations(
    frame_files,
    candidate_masks_per_frame,
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

        candidate_masks = [candidate["mask"] for candidate in candidate_masks_per_frame[index]]
        candidate_labels = [
            (
                f"cand={candidate_index + 1} s={candidate['score']:.2f} "
                f"iou={candidate['coarse_iou']:.2f} ar={candidate['area_ratio']:.2f}"
            )
            for candidate_index, candidate in enumerate(candidate_masks_per_frame[index])
        ]

        if visualization_cfg.get("save_object_overlays", True):
            object_overlay_frame = render_instance_mask_overlay(
                frame,
                candidate_masks,
                instance_labels=candidate_labels,
                overlay_alpha=overlay_alpha,
                title="SAM3 candidates",
            )
            object_output_path = os.path.join(
                visualization_dirs["motion_scores"],
                build_frame_filename(common_cfg, index, artifact="frame"),
            )
            save_visualization_frame(object_overlay_frame, object_output_path)

        mask_overlay_frame = render_mask_overlay(frame, processed_masks[index], overlay_alpha=overlay_alpha)
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
    visualization_cfg = pipeline_cfg.get("visualization", {})
    postprocess_cfg = pipeline_cfg.get("postprocess", {})
    seg_cfg = phase_cfg.get("models", {}).get("segmentation_upgrade", {})
    refinement_cfg = pipeline_cfg.get("refinement", {})
    inpainting_cfg = phase_cfg.get("models", {}).get("inpainting", {})

    input_frames_dir = sequence_spec["frames_dir"]
    frame_files = collect_image_paths(input_frames_dir, common_cfg)
    if args.max_frames is not None:
        frame_files = frame_files[: args.max_frames]
    if not frame_files:
        raise FileNotFoundError(f"No frames found for sequence: {input_frames_dir}")

    coarse_masks_dir = resolve_baseline_mask_dir(common_cfg, phase_cfg, sequence_spec["output_name"])
    baseline_objects_dir = resolve_baseline_object_mask_dir(common_cfg, phase_cfg, sequence_spec["output_name"])
    if not os.path.isdir(coarse_masks_dir) and not os.path.isdir(baseline_objects_dir):
        raise FileNotFoundError(
            "Baseline masks are unavailable. "
            f"Expected combined masks under {coarse_masks_dir} or object masks under {baseline_objects_dir}."
        )

    print(f"Sequence: {sequence_spec['output_name']}")
    print(f"Frames: {input_frames_dir}")
    print(f"Combined baseline masks: {coarse_masks_dir}")
    print(f"Object baseline masks: {baseline_objects_dir}")

    baseline_object_masks = load_object_masks(frame_files, baseline_objects_dir)
    mask_source = resolve_baseline_mask_source(phase_cfg, baseline_object_masks)
    if mask_source == "objects_union":
        coarse_masks = build_union_masks(baseline_object_masks, frame_files)
    else:
        coarse_masks = load_coarse_masks(frame_files, coarse_masks_dir, common_cfg)
    print(f"Baseline mask source: {mask_source}")

    refiner = SAM3UpgradeRefiner(
        weights_dir=seg_cfg.get("weights_dir", "models/sam3"),
        checkpoint_dir=seg_cfg.get("checkpoint_dir", "models/sam3/checkpoints"),
        checkpoint_name=seg_cfg.get("checkpoint_name"),
        repo_id=seg_cfg.get("repo_id"),
        auto_download=seg_cfg.get("auto_download", True),
        confidence_threshold=refinement_cfg.get("confidence_threshold", 0.35),
        box_expand_ratio=refinement_cfg.get("box_expand_ratio", 0.08),
        min_mask_area=refinement_cfg.get("min_mask_area", 32),
        min_selection_iou=refinement_cfg.get("min_selection_iou", 0.05),
        min_area_ratio=refinement_cfg.get("min_area_ratio", 0.95),
        max_area_ratio=refinement_cfg.get("max_area_ratio", 1.05),
        min_candidate_precision=refinement_cfg.get("min_candidate_precision", 0.95),
        min_coarse_recall=refinement_cfg.get("min_coarse_recall", 0.95),
        fallback_to_coarse=refinement_cfg.get("fallback_to_coarse", True),
    )

    print("Refining coarse masks with SAM3...")
    candidate_masks_per_frame, raw_refined_masks = refiner.refine(frame_files, coarse_masks)
    skip_postprocess = mask_source == "objects_union" and postprocess_cfg.get("skip_for_objects_union", False)
    if skip_postprocess:
        processed_masks = normalize_masks(raw_refined_masks)
    else:
        processed_masks = postprocess_masks(raw_refined_masks, postprocess_cfg)

    masks_root = get_phase_output_dir(phase_cfg, "masks_dir") or "results/masks/part3"
    sequence_masks_dir = os.path.join(masks_root, sequence_spec["output_name"])
    object_masks_dir = os.path.join(sequence_masks_dir, "objects")
    combined_masks_dir = os.path.join(sequence_masks_dir, "combined")

    try:
        from part3_exploration.diffusion_controlnet import ControlNetInpainter
    except ModuleNotFoundError:
        try:
            from diffusion_controlnet import ControlNetInpainter
        except ModuleNotFoundError:
            pass

    inpainting_cfg = phase_cfg.get("models", {}).get("inpainting", {})
    diffusion_cfg = phase_cfg.get("models", {}).get("refinement", {})

    if pipeline_cfg.get("export", {}).get("save_masks", True):
        save_candidate_masks(candidate_masks_per_frame, object_masks_dir)
        save_combined_masks(processed_masks, frame_files, combined_masks_dir)

    # ---------------------------------------------------------
    # Direction C: Generative Inpainting for Keyframes 
    # ---------------------------------------------------------
    # We load the frames to pass to SD
    frames_rgb = [read_image(f, mode="RGB") for f in frame_files]
    mid_idx = len(frames_rgb) // 2
    keyframe_indices = [0, mid_idx] if len(frames_rgb) > 1 else [0]
    
    print("\n--- Running SD Inpainting on Keyframes ---")
    sd_inpainter = ControlNetInpainter(weights_dir=diffusion_cfg.get("weights_dir", "models/stable-diffusion-inpainting"))
    generated_keyframes = sd_inpainter.inpaint(frames_rgb, processed_masks, keyframe_indices=keyframe_indices)
    
    # Save generated keyframes to results/masks/part3/keyframes
    keyframes_dir = os.path.join(sequence_masks_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    for idx, generated_img in generated_keyframes.items():
        out_path = os.path.join(keyframes_dir, f"frame_{idx:04d}_sd_inpainted.png")
        Image.fromarray(generated_img).save(out_path)
        print(f"Saved SD inpainted keyframe to {out_path}")
    print("------------------------------------------\n")
    
    # Pass to ProPainter
    inpainter = ProPainterInpainter(inpainting_cfg.get("weights_dir", "models/ProPainter"))
    output_video_dir = get_phase_output_dir(phase_cfg, "videos_dir") or "results/videos/part3"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(
        output_video_dir,
        build_video_filename(common_cfg, sequence_spec["output_name"], phase_cfg.get("phase", {}).get("slug", "part3")),
    )

    print("Inpainting with ProPainter...")
    restored_frames = inpainter.inpaint(input_frames_dir, combined_masks_dir, output_video_path)

    visualization_root = get_phase_output_dir(phase_cfg, "visualizations_dir") or "results/visualizations/part3"
    export_visualizations(
        frame_files,
        candidate_masks_per_frame,
        processed_masks,
        restored_frames,
        visualization_root,
        sequence_spec["output_name"],
        common_cfg,
        visualization_cfg,
    )

    print("Part 3 pipeline finished.")
    print(f"Saved object masks to: {object_masks_dir}")
    print(f"Saved combined masks to: {combined_masks_dir}")
    print(f"Saved inpainted video to: {output_video_path}")


if __name__ == "__main__":
    main()