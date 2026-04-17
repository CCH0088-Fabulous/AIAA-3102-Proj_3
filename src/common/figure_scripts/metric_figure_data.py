from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.metrics import build_background_valid_mask, compute_iou, compute_psnr, compute_ssim

try:
    from .metric_figure_constants import PAIR_ORDER, PHASE_LABELS, PHASE_ORDER, sequence_label
except ImportError:
    from metric_figure_constants import PAIR_ORDER, PHASE_LABELS, PHASE_ORDER, sequence_label


WILD_SEQUENCE_NAME = "wild_video_frames"


def _list_image_paths(directory):
    return sorted(
        [
            path
            for pattern in ("*.png", "*.jpg", "*.jpeg")
            for path in Path(directory).glob(pattern)
        ]
    )


def _read_rgb_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask_image(mask_path):
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_path}")
    return image


def _align_frame(frame, reference_shape):
    if frame.shape[:2] == reference_shape[:2]:
        return frame
    return cv2.resize(frame, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_LINEAR)


def _align_mask(mask, reference_shape):
    if mask.shape[:2] == reference_shape[:2]:
        return mask
    return cv2.resize(mask, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_NEAREST)


def _load_video_frames(video_path, frame_count):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frames = []
    while len(frames) < frame_count:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

    if len(frames) < frame_count:
        raise ValueError(f"Video {video_path} only yielded {len(frames)} frames, expected at least {frame_count}")
    return frames


def _build_consensus_mask(mask_triplet):
    stacked = np.stack([(mask > 0).astype(np.uint8) for mask in mask_triplet], axis=0)
    return (stacked.sum(axis=0) >= 2).astype(np.uint8) * 255


def _summarize_mean(values):
    numeric = pd.Series(values, dtype=float)
    return float(numeric.mean()) if not numeric.empty else float("nan")


def ensure_wild_video_metrics(metrics_root):
    repo_root = Path(__file__).resolve().parents[3]
    raw_dir = repo_root / "data" / "raw" / WILD_SEQUENCE_NAME
    phase_assets = {
        "part1": {
            "mask_dir": repo_root / "results" / "masks" / "part1" / WILD_SEQUENCE_NAME,
            "video_path": repo_root / "results" / "videos" / "part1" / "wild_video_frames_part1.mp4",
        },
        "part2": {
            "mask_dir": repo_root / "results" / "masks" / "part2" / WILD_SEQUENCE_NAME / "combined",
            "video_path": repo_root / "results" / "videos" / "part2" / "wild_video_frames_inpainted.mp4",
        },
        "part3": {
            "mask_dir": repo_root / "results" / "masks" / "part3" / WILD_SEQUENCE_NAME / "combined",
            "video_path": repo_root / "results" / "videos" / "part3" / "wild_video_frames_part3.mp4",
        },
    }

    if not raw_dir.is_dir():
        return
    if any(not asset["mask_dir"].is_dir() or not asset["video_path"].is_file() for asset in phase_assets.values()):
        return

    source_paths = _list_image_paths(raw_dir)
    phase_mask_paths = {phase: _list_image_paths(asset["mask_dir"]) for phase, asset in phase_assets.items()}
    frame_count = min([len(source_paths)] + [len(paths) for paths in phase_mask_paths.values()])
    if frame_count == 0:
        return

    source_paths = source_paths[:frame_count]
    source_frames = [_read_rgb_image(path) for path in source_paths]
    phase_video_frames = {
        phase: _load_video_frames(asset["video_path"], frame_count) for phase, asset in phase_assets.items()
    }
    phase_masks = {
        phase: [_read_mask_image(path) for path in paths[:frame_count]] for phase, paths in phase_mask_paths.items()
    }

    for frame_index in range(frame_count):
        reference_shape = source_frames[frame_index].shape
        for phase in PHASE_ORDER:
            phase_video_frames[phase][frame_index] = _align_frame(phase_video_frames[phase][frame_index], reference_shape)
            phase_masks[phase][frame_index] = _align_mask(phase_masks[phase][frame_index], reference_shape)

    consensus_masks = [
        _build_consensus_mask([phase_masks[phase][frame_index] for phase in PHASE_ORDER])
        for frame_index in range(frame_count)
    ]

    metrics_root = Path(metrics_root)
    for phase in PHASE_ORDER:
        phase_output_dir = metrics_root / phase / WILD_SEQUENCE_NAME
        phase_output_dir.mkdir(parents=True, exist_ok=True)

        iou_rows = []
        quality_rows = []
        iou_values = []
        psnr_values = []
        ssim_values = []

        for frame_index in range(frame_count):
            predicted_mask = phase_masks[phase][frame_index]
            reference_mask = consensus_masks[frame_index]
            iou_stats = compute_iou(predicted_mask, reference_mask)
            iou_values.append(iou_stats["iou"])
            iou_rows.append(
                {
                    "phase": phase,
                    "sequence": WILD_SEQUENCE_NAME,
                    "frame_index": frame_index,
                    "predicted_file": phase_mask_paths[phase][frame_index].name,
                    "reference_file": "phase_consensus_majority_vote",
                    "iou": iou_stats["iou"],
                    "intersection": iou_stats["intersection"],
                    "union": iou_stats["union"],
                }
            )

            valid_mask = build_background_valid_mask(predicted_mask, None)
            psnr_value = compute_psnr(source_frames[frame_index], phase_video_frames[phase][frame_index], valid_mask=valid_mask)
            ssim_value = compute_ssim(source_frames[frame_index], phase_video_frames[phase][frame_index], valid_mask=valid_mask)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            quality_rows.append(
                {
                    "phase": phase,
                    "sequence": WILD_SEQUENCE_NAME,
                    "frame_index": frame_index,
                    "evaluation_mode": "background_preservation",
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "valid_pixels": int(np.count_nonzero(valid_mask)),
                }
            )

        iou_rows.append(
            {
                "phase": phase,
                "sequence": WILD_SEQUENCE_NAME,
                "frame_index": "mean",
                "predicted_file": frame_count,
                "reference_file": "phase_consensus_majority_vote",
                "iou": _summarize_mean(iou_values),
                "intersection": "",
                "union": "",
            }
        )
        quality_rows.append(
            {
                "phase": phase,
                "sequence": WILD_SEQUENCE_NAME,
                "frame_index": "mean",
                "evaluation_mode": "background_preservation",
                "psnr": _summarize_mean(psnr_values),
                "ssim": _summarize_mean(ssim_values),
                "valid_pixels": "",
            }
        )

        pd.DataFrame(iou_rows).to_csv(phase_output_dir / "iou_results.csv", index=False)
        pd.DataFrame(quality_rows).to_csv(phase_output_dir / "psnr_ssim.csv", index=False)


def load_metric_data(metrics_root):
    ensure_wild_video_metrics(metrics_root)

    iou_frames = []
    quality_frames = []

    for csv_path in sorted(metrics_root.glob("*/*/*.csv")):
        phase = csv_path.parent.parent.name
        sequence = csv_path.parent.name
        metric_name = csv_path.stem

        frame_df = pd.read_csv(csv_path)
        frame_df = frame_df[frame_df["frame_index"].astype(str) != "mean"].copy()
        if frame_df.empty:
            continue

        frame_df["frame_index"] = pd.to_numeric(frame_df["frame_index"], errors="coerce").astype(int)
        frame_df["phase"] = phase
        frame_df["phase_label"] = PHASE_LABELS[phase]
        frame_df["sequence"] = sequence
        frame_df["sequence_label"] = sequence_label(sequence)

        if metric_name == "iou_results":
            frame_df["iou"] = pd.to_numeric(frame_df["iou"], errors="coerce")
            frame_df["intersection"] = pd.to_numeric(frame_df["intersection"], errors="coerce")
            frame_df["union"] = pd.to_numeric(frame_df["union"], errors="coerce")
            iou_frames.append(frame_df)
        elif metric_name == "psnr_ssim":
            frame_df["psnr"] = pd.to_numeric(frame_df["psnr"], errors="coerce")
            frame_df["ssim"] = pd.to_numeric(frame_df["ssim"], errors="coerce")
            frame_df["valid_pixels"] = pd.to_numeric(frame_df["valid_pixels"], errors="coerce")
            quality_frames.append(frame_df)

    if not iou_frames or not quality_frames:
        raise FileNotFoundError(
            f"Unable to find complete metric CSV sets under {metrics_root}. Expected both IoU and PSNR/SSIM files."
        )

    iou_df = pd.concat(iou_frames, ignore_index=True)
    quality_df = pd.concat(quality_frames, ignore_index=True)

    sequence_order = sorted(set(iou_df["sequence"]).union(quality_df["sequence"]))
    iou_df["phase"] = pd.Categorical(iou_df["phase"], categories=PHASE_ORDER, ordered=True)
    quality_df["phase"] = pd.Categorical(quality_df["phase"], categories=PHASE_ORDER, ordered=True)
    iou_df["sequence"] = pd.Categorical(iou_df["sequence"], categories=sequence_order, ordered=True)
    quality_df["sequence"] = pd.Categorical(quality_df["sequence"], categories=sequence_order, ordered=True)

    return iou_df.sort_values(["sequence", "phase", "frame_index"]), quality_df.sort_values(
        ["sequence", "phase", "frame_index"]
    )


def compute_summaries(iou_df, quality_df):
    iou_summary = (
        iou_df.groupby(["sequence", "phase", "sequence_label", "phase_label"], observed=True)
        .agg(
            n_frames=("iou", "size"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            std_iou=("iou", "std"),
            q1_iou=("iou", lambda series: series.quantile(0.25)),
            q3_iou=("iou", lambda series: series.quantile(0.75)),
            min_iou=("iou", "min"),
            max_iou=("iou", "max"),
            sum_intersection=("intersection", "sum"),
            sum_union=("union", "sum"),
        )
        .reset_index()
    )
    iou_summary["weighted_iou"] = iou_summary["sum_intersection"] / iou_summary["sum_union"].replace(0, np.nan)

    quality_summary = (
        quality_df.groupby(
            ["sequence", "phase", "sequence_label", "phase_label", "evaluation_mode"],
            observed=True,
        )
        .agg(
            mean_psnr=("psnr", "mean"),
            median_psnr=("psnr", "median"),
            std_psnr=("psnr", "std"),
            q1_psnr=("psnr", lambda series: series.quantile(0.25)),
            q3_psnr=("psnr", lambda series: series.quantile(0.75)),
            mean_ssim=("ssim", "mean"),
            median_ssim=("ssim", "median"),
            std_ssim=("ssim", "std"),
            q1_ssim=("ssim", lambda series: series.quantile(0.25)),
            q3_ssim=("ssim", lambda series: series.quantile(0.75)),
            mean_valid_pixels=("valid_pixels", "mean"),
            min_valid_pixels=("valid_pixels", "min"),
            max_valid_pixels=("valid_pixels", "max"),
        )
        .reset_index()
    )

    summary = iou_summary.merge(
        quality_summary,
        on=["sequence", "phase", "sequence_label", "phase_label"],
        how="outer",
    ).sort_values(["sequence", "phase"])
    return summary


def compute_paired_deltas(iou_df, quality_df):
    merged = iou_df[["sequence", "phase", "frame_index", "iou", "union"]].merge(
        quality_df[["sequence", "phase", "frame_index", "psnr", "ssim", "valid_pixels"]],
        on=["sequence", "phase", "frame_index"],
        how="inner",
    )

    delta_records = []
    for sequence in merged["sequence"].cat.categories:
        sequence_df = merged[merged["sequence"] == sequence]
        if sequence_df.empty:
            continue
        per_phase = {phase: phase_df.set_index("frame_index") for phase, phase_df in sequence_df.groupby("phase")}
        for phase_a, phase_b in PAIR_ORDER:
            if phase_a not in per_phase or phase_b not in per_phase:
                continue
            common_index = per_phase[phase_a].index.intersection(per_phase[phase_b].index)
            if len(common_index) == 0:
                continue
            first = per_phase[phase_a].loc[common_index]
            second = per_phase[phase_b].loc[common_index]
            for frame_index in common_index:
                delta_records.append(
                    {
                        "sequence": sequence,
                        "sequence_label": sequence_label(sequence),
                        "phase_from": phase_a,
                        "phase_to": phase_b,
                        "pair_label": f"{PHASE_LABELS[phase_a]} -> {PHASE_LABELS[phase_b]}",
                        "frame_index": int(frame_index),
                        "delta_iou": float(second.loc[frame_index, "iou"] - first.loc[frame_index, "iou"]),
                        "delta_psnr": float(second.loc[frame_index, "psnr"] - first.loc[frame_index, "psnr"]),
                        "delta_ssim": float(second.loc[frame_index, "ssim"] - first.loc[frame_index, "ssim"]),
                    }
                )

    deltas_df = pd.DataFrame(delta_records)
    if deltas_df.empty:
        raise ValueError("No paired metric deltas could be computed from the current metric files.")

    paired_summary_rows = []
    for (sequence, phase_from, phase_to, pair_label), subset in deltas_df.groupby(
        ["sequence", "phase_from", "phase_to", "pair_label"], observed=True
    ):
        row = {
            "sequence": sequence,
            "sequence_label": sequence_label(sequence),
            "phase_from": phase_from,
            "phase_to": phase_to,
            "pair_label": pair_label,
            "n_frames": int(len(subset)),
        }
        for delta_col in ["delta_iou", "delta_psnr", "delta_ssim"]:
            values = subset[delta_col]
            metric_suffix = delta_col.replace("delta_", "")
            row[f"mean_{metric_suffix}"] = float(values.mean())
            row[f"median_{metric_suffix}"] = float(values.median())
            row[f"std_{metric_suffix}"] = float(values.std())
            row[f"improved_{metric_suffix}"] = int((values > 0).sum())
            row[f"worsened_{metric_suffix}"] = int((values < 0).sum())
            row[f"equal_{metric_suffix}"] = int((values == 0).sum())
        paired_summary_rows.append(row)

    paired_summary = pd.DataFrame(paired_summary_rows).sort_values(["sequence", "phase_from", "phase_to"])
    return deltas_df, paired_summary


def normalize_columns(dataframe, columns):
    normalized = dataframe[columns].copy()
    for column in columns:
        col_min = normalized[column].min()
        col_max = normalized[column].max()
        if pd.isna(col_min) or pd.isna(col_max) or np.isclose(col_min, col_max):
            normalized[column] = 0.5
        else:
            normalized[column] = (normalized[column] - col_min) / (col_max - col_min)
    return normalized