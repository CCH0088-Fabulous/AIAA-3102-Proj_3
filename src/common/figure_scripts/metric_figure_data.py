import numpy as np
import pandas as pd

try:
    from .metric_figure_constants import PAIR_ORDER, PHASE_LABELS, PHASE_ORDER, sequence_label
except ImportError:
    from metric_figure_constants import PAIR_ORDER, PHASE_LABELS, PHASE_ORDER, sequence_label


def load_metric_data(metrics_root):
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