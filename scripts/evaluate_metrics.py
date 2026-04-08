import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.config import (
    build_video_filename,
    collect_image_paths,
    ensure_phase_output_dirs,
    get_phase_output_dir,
    load_yaml_config,
    resolve_sequence_spec,
)
from common.metrics import build_background_valid_mask, compute_iou, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mask IoU and video restoration quality.")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument(
        "--phase-config",
        default="configs/part1_baseline.yaml",
        help="Phase config whose outputs should be evaluated.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence key from configs/common.yaml or a direct folder containing frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on how many frames to evaluate.",
    )
    parser.add_argument(
        "--reference-frames-dir",
        default=None,
        help="Optional clean reference frame directory for full-reference PSNR/SSIM.",
    )
    parser.add_argument(
        "--reference-mask-dir",
        default=None,
        help="Optional reference mask directory. Defaults to dataset-specific masks when available.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Optional predicted mask directory. Defaults to the phase output mask directory for the sequence.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional restored video path. Defaults to the phase output video path for the sequence.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_reference_mask_dir(args, common_cfg, sequence_spec):
    if args.reference_mask_dir:
        return args.reference_mask_dir

    reference_mask_dir = sequence_spec.get("reference_masks_dir")
    if reference_mask_dir and os.path.isdir(reference_mask_dir):
        return reference_mask_dir

    davis_annotation_root = common_cfg.get("datasets", {}).get("davis", {}).get("annotation_root")
    if davis_annotation_root:
        candidate = os.path.join(davis_annotation_root, sequence_spec["output_name"])
        if os.path.isdir(candidate):
            return candidate
    return None


def load_video_frames(video_path, max_frames=None):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Restored video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return frames


def read_image(path, mode="RGB"):
    try:
        image = Image.open(path)
    except Exception as exc:
        raise ValueError(f"Failed to read image: {path}") from exc

    if mode == "RGB":
        image = image.convert("RGB")
    elif mode == "L":
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    return np.array(image)


def resolve_mask_directory(mask_dir, common_cfg):
    if not mask_dir or not os.path.isdir(mask_dir):
        return mask_dir

    image_paths = collect_image_paths(mask_dir, common_cfg)
    if image_paths:
        return mask_dir

    combined_dir = os.path.join(mask_dir, "combined")
    if os.path.isdir(combined_dir):
        combined_paths = collect_image_paths(combined_dir, common_cfg)
        if combined_paths:
            return combined_dir

    return mask_dir


def load_images_from_dir(directory, common_cfg, max_frames=None, mode="RGB"):
    if not directory or not os.path.isdir(directory):
        return [], []
    image_paths = collect_image_paths(directory, common_cfg)
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    return [read_image(path, mode=mode) for path in image_paths], image_paths


def align_frame(prediction, reference):
    if prediction.shape[:2] == reference.shape[:2]:
        return prediction
    return np.array(
        Image.fromarray(prediction).resize(
            (reference.shape[1], reference.shape[0]), resample=Image.BILINEAR
        )
    )


def align_mask(prediction, reference):
    if prediction.shape[:2] == reference.shape[:2]:
        return prediction
    return np.array(
        Image.fromarray(prediction).resize(
            (reference.shape[1], reference.shape[0]), resample=Image.NEAREST
        )
    )


def summarize_numeric(values):
    valid_values = [value for value in values if not math.isnan(value)]
    if not valid_values:
        return float("nan")
    return float(sum(valid_values) / len(valid_values))


def upsert_csv_rows(file_path, fieldnames, key_fields, rows):
    existing_rows = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8", newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_rows = list(reader)

    row_map = {
        tuple(existing_row.get(field, "") for field in key_fields): existing_row
        for existing_row in existing_rows
    }
    for row in rows:
        row_map[tuple(str(row.get(field, "")) for field in key_fields)] = {
            field: row.get(field, "")
            for field in fieldnames
        }

    with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(row_map):
            writer.writerow(row_map[key])


def format_metric_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def evaluate_mask_iou(predicted_mask_dir, reference_mask_dir, common_cfg, phase_name, sequence_name, max_frames=None):
    predicted_paths = collect_image_paths(predicted_mask_dir, common_cfg)
    reference_paths = collect_image_paths(reference_mask_dir, common_cfg)
    if max_frames is not None:
        predicted_paths = predicted_paths[:max_frames]
        reference_paths = reference_paths[:max_frames]

    paired_count = min(len(predicted_paths), len(reference_paths))
    predicted_paths = predicted_paths[:paired_count]
    reference_paths = reference_paths[:paired_count]

    rows = []
    iou_scores = []
    for frame_index, (predicted_path, reference_path) in enumerate(zip(predicted_paths, reference_paths)):
        predicted_mask = read_image(predicted_path, mode="L")
        reference_mask = read_image(reference_path, mode="L")
        predicted_mask = align_mask(predicted_mask, reference_mask)

        iou_stats = compute_iou(predicted_mask, reference_mask)
        iou_scores.append(iou_stats["iou"])
        rows.append(
            {
                "phase": phase_name,
                "sequence": sequence_name,
                "frame_index": frame_index,
                "predicted_file": os.path.basename(predicted_path),
                "reference_file": os.path.basename(reference_path),
                "iou": format_metric_value(iou_stats["iou"]),
                "intersection": iou_stats["intersection"],
                "union": iou_stats["union"],
            }
        )

    summary_row = {
        "phase": phase_name,
        "sequence": sequence_name,
        "frame_index": "mean",
        "predicted_file": paired_count,
        "reference_file": paired_count,
        "iou": format_metric_value(summarize_numeric(iou_scores)),
        "intersection": "",
        "union": "",
    }
    rows.append(summary_row)
    return rows, summary_row


def evaluate_video_quality(
    restored_frames,
    reference_frames,
    predicted_masks,
    reference_masks,
    phase_name,
    sequence_name,
    evaluation_mode,
):
    paired_count = min(len(restored_frames), len(reference_frames))
    rows = []
    psnr_scores = []
    ssim_scores = []

    for frame_index in range(paired_count):
        restored = restored_frames[frame_index]
        reference = reference_frames[frame_index]
        restored = align_frame(restored, reference)

        valid_mask = None
        if evaluation_mode == "background_preservation":
            predicted_mask = predicted_masks[frame_index] if frame_index < len(predicted_masks) else None
            reference_mask = reference_masks[frame_index] if frame_index < len(reference_masks) else None
            if predicted_mask is not None:
                predicted_mask = align_mask(predicted_mask, reference[..., 0] if reference.ndim == 3 else reference)
            if reference_mask is not None:
                reference_mask = align_mask(reference_mask, reference[..., 0] if reference.ndim == 3 else reference)
            if predicted_mask is not None or reference_mask is not None:
                valid_mask = build_background_valid_mask(predicted_mask, reference_mask)

        psnr_value = compute_psnr(reference, restored, valid_mask=valid_mask)
        ssim_value = compute_ssim(reference, restored, valid_mask=valid_mask)
        psnr_scores.append(psnr_value)
        ssim_scores.append(ssim_value)

        valid_pixels = ""
        if valid_mask is not None:
            valid_pixels = int(np.count_nonzero(valid_mask))

        rows.append(
            {
                "phase": phase_name,
                "sequence": sequence_name,
                "frame_index": frame_index,
                "evaluation_mode": evaluation_mode,
                "psnr": format_metric_value(psnr_value),
                "ssim": format_metric_value(ssim_value),
                "valid_pixels": valid_pixels,
            }
        )

    summary_row = {
        "phase": phase_name,
        "sequence": sequence_name,
        "frame_index": "mean",
        "evaluation_mode": evaluation_mode,
        "psnr": format_metric_value(summarize_numeric(psnr_scores)),
        "ssim": format_metric_value(summarize_numeric(ssim_scores)),
        "valid_pixels": "",
    }
    rows.append(summary_row)
    return rows, summary_row


def main():
    args = parse_args()
    common_cfg = load_yaml_config(args.common_config)
    phase_cfg = load_yaml_config(args.phase_config)
    ensure_phase_output_dirs(phase_cfg)

    phase_name = phase_cfg.get("phase", {}).get("slug", phase_cfg.get("phase", {}).get("name", "phase"))
    metrics_root = common_cfg.get("paths", {}).get("metrics_root", "results/metrics")
    phase_metrics_root = os.path.join(metrics_root, phase_name)
    ensure_dir(phase_metrics_root)

    default_sequence = phase_cfg.get("pipeline", {}).get("input", {}).get(
        "sequence_key",
        common_cfg.get("project", {}).get("default_sequence", "bmx-trees"),
    )
    sequence_spec = resolve_sequence_spec(args.sequence or default_sequence, common_cfg)
    sequence_name = sequence_spec["output_name"]
    sequence_metrics_root = os.path.join(phase_metrics_root, sequence_name)
    ensure_dir(sequence_metrics_root)
    sequence_spec = resolve_sequence_spec(args.sequence or default_sequence, common_cfg)
    sequence_name = sequence_spec["output_name"]

    predicted_mask_dir = args.mask_dir or os.path.join(
        get_phase_output_dir(phase_cfg, "masks_dir"),
        sequence_name,
    )
    predicted_mask_dir = resolve_mask_directory(predicted_mask_dir, common_cfg)
    output_video_dir = get_phase_output_dir(phase_cfg, "videos_dir")
    default_video_path = os.path.join(
        output_video_dir,
        build_video_filename(common_cfg, sequence_name, phase_name),
    )
    predicted_video_path = args.video_path or default_video_path
    if not os.path.isfile(predicted_video_path):
        alternative_video_path = os.path.join(output_video_dir, f"{sequence_name}_inpainted.mp4")
        if os.path.isfile(alternative_video_path):
            predicted_video_path = alternative_video_path
        else:
            wildcard = os.path.join(output_video_dir, f"{sequence_name}*.mp4")
            candidates = sorted(glob.glob(wildcard))
            if candidates:
                predicted_video_path = candidates[-1]
    reference_mask_dir = resolve_reference_mask_dir(args, common_cfg, sequence_spec)

    print(f"Evaluating phase: {phase_name}")
    print(f"Sequence: {sequence_name}")
    print(f"Metrics root: {metrics_root}")

    iou_summary = None
    if reference_mask_dir and os.path.isdir(predicted_mask_dir):
        iou_rows, iou_summary = evaluate_mask_iou(
            predicted_mask_dir,
            reference_mask_dir,
            common_cfg,
            phase_name,
            sequence_name,
            max_frames=args.max_frames,
        )
        iou_csv_path = os.path.join(sequence_metrics_root, "iou_results.csv")
        upsert_csv_rows(
            iou_csv_path,
            ["phase", "sequence", "frame_index", "predicted_file", "reference_file", "iou", "intersection", "union"],
            ["phase", "sequence", "frame_index"],
            iou_rows,
        )
        print(f"Saved IoU results to {iou_csv_path}")
    else:
        print("Skipping IoU evaluation because reference masks or predicted masks are unavailable.")

    restored_frames = load_video_frames(predicted_video_path, max_frames=args.max_frames)
    reference_dir = args.reference_frames_dir or sequence_spec["frames_dir"]
    reference_frames, reference_paths = load_images_from_dir(
        reference_dir,
        common_cfg,
        max_frames=args.max_frames,
        mode="RGB",
    )
    predicted_mask_frames, _ = load_images_from_dir(
        predicted_mask_dir,
        common_cfg,
        max_frames=args.max_frames,
        mode="L",
    )
    reference_mask_frames = []
    if reference_mask_dir:
        reference_mask_frames, _ = load_images_from_dir(
            reference_mask_dir,
            common_cfg,
            max_frames=args.max_frames,
            mode="L",
        )

    evaluation_mode = "full_reference" if args.reference_frames_dir else "background_preservation"
    psnr_ssim_rows, quality_summary = evaluate_video_quality(
        restored_frames,
        reference_frames,
        predicted_mask_frames,
        reference_mask_frames,
        phase_name,
        sequence_name,
        evaluation_mode,
    )
    psnr_ssim_csv_path = os.path.join(sequence_metrics_root, "psnr_ssim.csv")
    upsert_csv_rows(
        psnr_ssim_csv_path,
        ["phase", "sequence", "frame_index", "evaluation_mode", "psnr", "ssim", "valid_pixels"],
        ["phase", "sequence", "evaluation_mode", "frame_index"],
        psnr_ssim_rows,
    )
    print(f"Saved PSNR/SSIM results to {psnr_ssim_csv_path}")

    if iou_summary is not None:
        print(f"Mean IoU: {iou_summary['iou']}")
    print(f"Mean PSNR ({evaluation_mode}): {quality_summary['psnr']}")
    print(f"Mean SSIM ({evaluation_mode}): {quality_summary['ssim']}")


if __name__ == "__main__":
    main()