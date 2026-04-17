from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROXY_IOU_NOTE = "Wild Video uses a phase-consensus proxy IoU because ground-truth masks are unavailable."

try:
    from .metric_figure_constants import (
        METRIC_LABELS,
        PAIR_ORDER,
        PHASE_COLORS,
        PHASE_LABELS,
        PHASE_ORDER,
        sequence_label,
    )
    from .metric_figure_data import normalize_columns
except ImportError:
    from metric_figure_constants import (
        METRIC_LABELS,
        PAIR_ORDER,
        PHASE_COLORS,
        PHASE_LABELS,
        PHASE_ORDER,
        sequence_label,
    )
    from metric_figure_data import normalize_columns


def set_plot_style():
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.facecolor": "#FAFAF7",
            "figure.facecolor": "#F6F3EE",
            "grid.color": "#D6D1C7",
            "grid.alpha": 0.45,
            "axes.edgecolor": "#443D36",
            "axes.labelcolor": "#2F2A26",
            "xtick.color": "#2F2A26",
            "ytick.color": "#2F2A26",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        },
    )
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 320


def save_figure(fig, output_dir, stem):
    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def finalize_figure_layout(fig, *, top=0.90, bottom=0.10, left=0.08, right=0.98, hspace=None, wspace=None):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)


def add_top_center_legend(fig, handles, labels, *, y=0.945, ncol=3):
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
    )


def apply_common_axis_style(ax):
    ax.grid(axis="y", alpha=0.35)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)


def add_jitter_points(ax, grouped_values, colors, rng):
    for position, (phase, values) in enumerate(grouped_values.items()):
        if len(values) == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.05, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            alpha=0.30,
            color=colors[phase],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )


def _present_sequences(dataframe):
    if hasattr(dataframe["sequence"], "cat"):
        return [sequence for sequence in dataframe["sequence"].cat.categories if (dataframe["sequence"] == sequence).any()]
    return list(pd.unique(dataframe["sequence"]))


def _contains_wild_sequence(dataframe):
    return any(str(sequence) == "wild_video_frames" for sequence in _present_sequences(dataframe))


def draw_distribution_panel(ax, subset, summary_subset, y_col, y_label, title, rng, y_limits=None):
    palette = [PHASE_COLORS[phase] for phase in PHASE_ORDER]
    sns.violinplot(
        data=subset,
        x="phase",
        hue="phase",
        y=y_col,
        order=PHASE_ORDER,
        hue_order=PHASE_ORDER,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0,
        saturation=1.0,
        dodge=False,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=subset,
        x="phase",
        y=y_col,
        order=PHASE_ORDER,
        width=0.24,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "#FFFDF9", "edgecolor": "#2F2A26", "linewidth": 1.2, "alpha": 0.9},
        whiskerprops={"color": "#2F2A26", "linewidth": 1.15},
        capprops={"color": "#2F2A26", "linewidth": 1.15},
        medianprops={"color": "#111111", "linewidth": 1.6},
        ax=ax,
    )

    grouped_values = {
        phase: subset.loc[subset["phase"] == phase, y_col].to_numpy() for phase in PHASE_ORDER
    }
    add_jitter_points(ax, grouped_values, PHASE_COLORS, rng)

    for position, phase in enumerate(PHASE_ORDER):
        row = summary_subset[summary_subset["phase"] == phase]
        if row.empty:
            continue
        mean_value = row.iloc[0][f"mean_{y_col}"] if f"mean_{y_col}" in row.columns else row.iloc[0][y_col]
        ax.scatter(
            position,
            mean_value,
            marker="D",
            s=80,
            color="#FFF6E8",
            edgecolor="#2F2A26",
            linewidth=1.0,
            zorder=4,
        )

    ax.set_title(title, fontsize=18, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(PHASE_ORDER)))
    ax.set_xticklabels([PHASE_LABELS[phase].replace("Part ", "P") for phase in PHASE_ORDER], rotation=10)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    apply_common_axis_style(ax)


def plot_iou_distribution(iou_df, summary_df, output_dir, stem="01_iou_distribution"):
    sequences = _present_sequences(iou_df)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(sequences), figsize=(7.2 * len(sequences), 6.4), sharey=True)
    if len(sequences) == 1:
        axes = [axes]
    for ax, sequence in zip(axes, sequences):
        subset = iou_df[iou_df["sequence"] == sequence]
        summary_subset = summary_df[summary_df["sequence"] == sequence]
        draw_distribution_panel(
            ax,
            subset,
            summary_subset,
            y_col="iou",
            y_label="Intersection over Union",
            title=sequence_label(sequence),
            rng=rng,
            y_limits=(0.0, 1.0),
        )
    fig.suptitle("Frame-Level IoU Distributions by Phase", fontsize=20, y=1.02)
    footer_lines = ["Violin density + box summary + frame-level jittered observations"]
    if _contains_wild_sequence(iou_df):
        footer_lines.append(PROXY_IOU_NOTE)
    fig.text(0.5, 0.02, "\n".join(footer_lines), ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.84, bottom=0.20 if _contains_wild_sequence(iou_df) else 0.16, wspace=0.14)
    save_figure(fig, output_dir, stem)


def plot_quality_distribution(quality_df, summary_df, output_dir, stem="02_quality_distribution"):
    sequences = _present_sequences(quality_df)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, len(sequences), figsize=(7.0 * len(sequences), 10.5), sharex=False)
    for col, sequence in enumerate(sequences):
        subset = quality_df[quality_df["sequence"] == sequence]
        summary_subset = summary_df[summary_df["sequence"] == sequence]
        draw_distribution_panel(
            axes[0, col],
            subset,
            summary_subset,
            y_col="psnr",
            y_label="PSNR (dB)",
            title=f"{sequence_label(sequence)}\nPSNR (background preservation)",
            rng=rng,
        )
        draw_distribution_panel(
            axes[1, col],
            subset,
            summary_subset,
            y_col="ssim",
            y_label="SSIM",
            title=f"{sequence_label(sequence)}\nSSIM (background preservation)",
            rng=rng,
        )
        axes[0, col].tick_params(labelbottom=False)
    fig.suptitle("Frame-Level Quality Metric Distributions", fontsize=20, y=1.01)
    fig.text(
        0.5,
        0.02,
        "All current PSNR/SSIM values use background_preservation mode, not full-reference restoration quality.",
        ha="center",
        fontsize=12,
    )
    finalize_figure_layout(fig, top=0.87, bottom=0.13, hspace=0.22, wspace=0.18)
    save_figure(fig, output_dir, stem)


def _framewise_stem(sequence):
    stems = {
        "bmx-trees": "03_framewise_metrics_bmx_trees",
        "tennis": "04_framewise_metrics_tennis",
        "wild_video_frames": "14_framewise_metrics_wild_video",
    }
    return stems.get(str(sequence), f"framewise_metrics_{str(sequence).replace('-', '_')}")


def _paired_delta_stem(sequence):
    stems = {
        "bmx-trees": "05_paired_deltas_bmx_trees",
        "tennis": "06_paired_deltas_tennis",
        "wild_video_frames": "15_paired_deltas_wild_videos",
    }
    return stems.get(str(sequence), f"paired_deltas_{str(sequence).replace('-', '_')}")


def plot_sequence_trends(iou_df, quality_df, output_dir, sequence):
    merged = iou_df[iou_df["sequence"] == sequence][["phase", "frame_index", "iou"]].merge(
        quality_df[quality_df["sequence"] == sequence][["phase", "frame_index", "psnr", "ssim"]],
        on=["phase", "frame_index"],
        how="inner",
    )
    metrics = ["iou", "psnr", "ssim"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    for ax, metric in zip(axes, metrics):
        for phase in PHASE_ORDER:
            phase_df = merged[merged["phase"] == phase].sort_values("frame_index")
            if phase_df.empty:
                continue
            rolling_mean = phase_df[metric].rolling(window=7, center=True, min_periods=1).mean()
            ax.plot(
                phase_df["frame_index"],
                phase_df[metric],
                color=PHASE_COLORS[phase],
                linewidth=1.0,
                alpha=0.18,
            )
            ax.plot(
                phase_df["frame_index"],
                rolling_mean,
                color=PHASE_COLORS[phase],
                linewidth=2.7,
                label=PHASE_LABELS[phase],
            )
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} across Frame Index", loc="left")
        if metric == "iou":
            ax.set_ylim(0.0, 1.0)
        apply_common_axis_style(ax)

    axes[0].legend(loc="upper right", ncol=3, frameon=True, fontsize=11)
    axes[-1].set_xlabel("Frame Index")
    fig.suptitle(f"Frame-Wise Metric Trends: {sequence_label(sequence)}", fontsize=20, y=1.01)
    if str(sequence) == "wild_video_frames":
        fig.text(0.5, 0.02, PROXY_IOU_NOTE, ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.92, bottom=0.12 if str(sequence) == "wild_video_frames" else 0.08, hspace=0.28)
    save_figure(fig, output_dir, _framewise_stem(sequence))


def plot_paired_deltas(deltas_df, output_dir, sequence):
    sequence_df = deltas_df[deltas_df["sequence"] == sequence].copy()
    metrics = ["delta_iou", "delta_psnr", "delta_ssim"]
    pair_labels = [f"{PHASE_LABELS[a]} -> {PHASE_LABELS[b]}" for a, b in PAIR_ORDER]
    fig, axes = plt.subplots(len(metrics), len(pair_labels), figsize=(18, 12))

    for row_idx, metric in enumerate(metrics):
        metric_name = metric.replace("delta_", "")
        for col_idx, pair_label in enumerate(pair_labels):
            ax = axes[row_idx, col_idx]
            subset = sequence_df[sequence_df["pair_label"] == pair_label]
            values = subset[metric].to_numpy()
            pair_to = subset["phase_to"].iloc[0] if not subset.empty else "part3"
            color = PHASE_COLORS[pair_to]
            if len(values) > 0:
                if np.unique(values).size > 1:
                    sns.histplot(
                        values,
                        bins=16,
                        kde=True,
                        stat="density",
                        color=color,
                        edgecolor="#FFFDF9",
                        linewidth=0.8,
                        alpha=0.90,
                        ax=ax,
                    )
                else:
                    ax.hist(values, bins=1, color=color, edgecolor="#FFFDF9", linewidth=0.8, alpha=0.90)
                ax.axvline(0.0, color="#1E1E1E", linestyle="--", linewidth=1.2)
                mean_delta = float(np.mean(values))
                median_delta = float(np.median(values))
                improved = int(np.sum(values > 0))
                worsened = int(np.sum(values < 0))
                equal = int(np.sum(values == 0))
                ax.text(
                    0.98,
                    0.94,
                    f"mean={mean_delta:+.3f}\nmedian={median_delta:+.3f}\n+ {improved} / - {worsened} / = {equal}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10.5,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFDF9", "edgecolor": "#C7C1B6", "alpha": 0.95},
                )
            ax.set_title(pair_label.replace("Part ", "P"), fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"Delta {METRIC_LABELS[metric_name]}")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Delta value")
            apply_common_axis_style(ax)

    fig.suptitle(f"Paired Frame-Level Improvements: {sequence_label(sequence)}", fontsize=20, y=1.01)
    if str(sequence) == "wild_video_frames":
        fig.text(0.5, 0.02, PROXY_IOU_NOTE, ha="center", fontsize=11.5)
    finalize_figure_layout(
        fig,
        top=0.91,
        bottom=0.12 if str(sequence) == "wild_video_frames" else 0.08,
        hspace=0.30,
        wspace=0.22,
    )
    save_figure(fig, output_dir, _paired_delta_stem(sequence))


def _load_rgb_image(image_path):
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Unable to read image frame: {image_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_rgb_video_frame(video_path, frame_index):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to read frame {frame_index} from video: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _select_representative_frame_indices(frame_count, rows=3):
    if frame_count <= 0:
        raise ValueError("wild_video_frames does not contain any source frames")
    if frame_count <= rows:
        return list(range(frame_count))
    sampled = np.linspace(0, frame_count - 1, num=rows + 2, dtype=int)[1:-1]
    return sampled.tolist()


def plot_wild_video_frames_comparison(output_dir, frame_indices=None):
    repo_root = Path(__file__).resolve().parents[3]
    raw_dir = repo_root / "data" / "raw" / "wild_video_frames"
    output_dir = Path(output_dir)
    phase_videos = {
        "part1": repo_root / "results" / "videos" / "part1" / "wild_video_frames_part1.mp4",
        "part2": repo_root / "results" / "videos" / "part2" / "wild_video_frames_inpainted.mp4",
        "part3": repo_root / "results" / "videos" / "part3" / "wild_video_frames_part3.mp4",
    }

    source_frames = sorted(raw_dir.glob("*.jpg"))
    if not source_frames:
        raise FileNotFoundError(f"No source frames found under {raw_dir}")
    missing_videos = [path for path in phase_videos.values() if not path.exists()]
    if missing_videos:
        missing_text = ", ".join(str(path) for path in missing_videos)
        raise FileNotFoundError(f"Missing wild_video_frames result videos: {missing_text}")

    selected_indices = frame_indices or _select_representative_frame_indices(len(source_frames), rows=3)
    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(16.5, 4.2 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Original", PHASE_LABELS["part1"], PHASE_LABELS["part2"], PHASE_LABELS["part3"]]
    column_keys = [None, "part1", "part2", "part3"]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=16, pad=12)

    for row_idx, frame_index in enumerate(selected_indices):
        if frame_index < 0 or frame_index >= len(source_frames):
            raise IndexError(f"Requested frame index {frame_index} is outside 0..{len(source_frames) - 1}")

        source_path = source_frames[frame_index]
        display_name = source_path.stem
        axes[row_idx, 0].imshow(_load_rgb_image(source_path))
        axes[row_idx, 0].text(
            -0.10,
            0.50,
            f"Frame {display_name}",
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            fontsize=12,
            rotation=90,
        )

        for col_idx, phase in enumerate(column_keys[1:], start=1):
            axes[row_idx, col_idx].imshow(_load_rgb_video_frame(phase_videos[phase], frame_index))

        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle("Qualitative Comparison on Wild Video Frames", fontsize=21, y=0.99)
    fig.text(
        0.5,
        0.02,
        "Representative early/mid/late frames from the unconstrained wild_video_frames sequence.",
        ha="center",
        fontsize=12,
    )
    finalize_figure_layout(fig, top=0.90, bottom=0.07, left=0.05, right=0.99, hspace=0.08, wspace=0.03)
    save_figure(fig, output_dir, "13_wild_video_frames_comparison")


def plot_iou_ecdf(iou_df, output_dir, stem="07_iou_ecdf"):
    sequences = _present_sequences(iou_df)
    fig, axes = plt.subplots(1, len(sequences), figsize=(7.2 * len(sequences), 6.2), sharey=True)
    if len(sequences) == 1:
        axes = [axes]
    for ax, sequence in zip(axes, sequences):
        subset = iou_df[iou_df["sequence"] == sequence]
        sns.ecdfplot(
            data=subset,
            x="iou",
            hue="phase",
            hue_order=PHASE_ORDER,
            palette=PHASE_COLORS,
            linewidth=2.5,
            ax=ax,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(sequence_label(sequence))
        ax.set_xlabel("IoU")
        ax.set_ylabel("Empirical CDF")
        apply_common_axis_style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    add_top_center_legend(fig, handles, [PHASE_LABELS.get(label, label) for label in labels], y=0.96, ncol=3)
    fig.suptitle("IoU Distribution Dominance via ECDF", fontsize=20, y=0.995)
    if _contains_wild_sequence(iou_df):
        fig.text(0.5, 0.02, PROXY_IOU_NOTE, ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.82, bottom=0.16 if _contains_wild_sequence(iou_df) else 0.11, wspace=0.18)
    save_figure(fig, output_dir, stem)


def fit_and_plot_line(ax, x_values, y_values, color):
    if len(x_values) < 2 or np.allclose(x_values, x_values[0]):
        return
    coefficients = np.polyfit(x_values, y_values, deg=1)
    x_range = np.linspace(np.min(x_values), np.max(x_values), 100)
    y_range = coefficients[0] * x_range + coefficients[1]
    ax.plot(x_range, y_range, color=color, linewidth=2.2, alpha=0.95)


def plot_iou_vs_union(iou_df, output_dir, stem="08_iou_vs_union"):
    sequences = _present_sequences(iou_df)
    fig, axes = plt.subplots(1, len(sequences), figsize=(7.5 * len(sequences), 6.3), sharey=True)
    if len(sequences) == 1:
        axes = [axes]

    for ax, sequence in zip(axes, sequences):
        subset = iou_df[iou_df["sequence"] == sequence]
        annotation_lines = []
        for phase in PHASE_ORDER:
            phase_df = subset[subset["phase"] == phase]
            if phase_df.empty:
                continue
            x_values = phase_df["union"].to_numpy(dtype=float)
            y_values = phase_df["iou"].to_numpy(dtype=float)
            ax.scatter(
                x_values,
                y_values,
                s=34,
                alpha=0.70,
                color=PHASE_COLORS[phase],
                edgecolors="white",
                linewidths=0.35,
                label=PHASE_LABELS[phase],
            )
            fit_and_plot_line(ax, x_values, y_values, PHASE_COLORS[phase])
            correlation = float(pd.Series(x_values).corr(pd.Series(y_values)))
            annotation_lines.append(f"{PHASE_LABELS[phase].replace('Part ', 'P')}: r={correlation:+.2f}")

        ax.text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFDF9", "edgecolor": "#C7C1B6", "alpha": 0.95},
        )
        ax.set_title(sequence_label(sequence))
        ax.set_xlabel("Union Pixels")
        ax.set_ylabel("IoU")
        ax.set_ylim(0.0, 1.0)
        apply_common_axis_style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    add_top_center_legend(fig, handles, labels, y=0.96, ncol=3)
    fig.suptitle("How Foreground Extent Relates to IoU", fontsize=20, y=0.995)
    footer_lines = ["Each point is one frame. Trend lines are simple linear fits."]
    if _contains_wild_sequence(iou_df):
        footer_lines.append(PROXY_IOU_NOTE)
    fig.text(0.5, 0.02, "\n".join(footer_lines), ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.82, bottom=0.20 if _contains_wild_sequence(iou_df) else 0.15, wspace=0.20)
    save_figure(fig, output_dir, stem)


def plot_quality_vs_valid_pixels(quality_df, output_dir, stem="09_quality_vs_valid_pixels"):
    sequences = _present_sequences(quality_df)
    metrics = ["psnr", "ssim"]
    fig, axes = plt.subplots(len(metrics), len(sequences), figsize=(7.4 * len(sequences), 10.8), sharex="col")

    for row_idx, metric in enumerate(metrics):
        for col_idx, sequence in enumerate(sequences):
            ax = axes[row_idx, col_idx]
            subset = quality_df[quality_df["sequence"] == sequence]
            annotation_lines = []
            for phase in PHASE_ORDER:
                phase_df = subset[subset["phase"] == phase]
                if phase_df.empty:
                    continue
                x_values = phase_df["valid_pixels"].to_numpy(dtype=float)
                y_values = phase_df[metric].to_numpy(dtype=float)
                ax.scatter(
                    x_values,
                    y_values,
                    s=34,
                    alpha=0.70,
                    color=PHASE_COLORS[phase],
                    edgecolors="white",
                    linewidths=0.35,
                    label=PHASE_LABELS[phase],
                )
                fit_and_plot_line(ax, x_values, y_values, PHASE_COLORS[phase])
                correlation = float(pd.Series(x_values).corr(pd.Series(y_values)))
                annotation_lines.append(f"{PHASE_LABELS[phase].replace('Part ', 'P')}: r={correlation:+.2f}")

            ax.text(
                0.02,
                0.98,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.0,
                bbox={"boxstyle": "round,pad=0.32", "facecolor": "#FFFDF9", "edgecolor": "#C7C1B6", "alpha": 0.95},
            )
            ax.set_title(f"{sequence_label(sequence)}: {METRIC_LABELS[metric]}", fontsize=18, pad=10)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Valid Background Pixels")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(METRIC_LABELS[metric])
            apply_common_axis_style(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    add_top_center_legend(fig, handles, labels, y=0.96, ncol=3)
    fig.suptitle("Quality Metrics versus Evaluated Background Area", fontsize=20, y=0.995)
    finalize_figure_layout(fig, top=0.82, bottom=0.09, hspace=0.10, wspace=0.20)
    save_figure(fig, output_dir, stem)


def plot_summary_heatmaps(summary_df, output_dir, stem="10_metric_summary_heatmaps"):
    display_df = summary_df.copy()
    display_df["row_label"] = display_df["sequence_label"] + " | " + display_df["phase_label"].str.replace("Part ", "P")
    mean_columns = ["mean_iou", "weighted_iou", "mean_psnr", "mean_ssim"]
    std_columns = ["std_iou", "std_psnr", "std_ssim"]

    mean_norm = normalize_columns(display_df, mean_columns)
    std_norm = normalize_columns(display_df, std_columns)
    mean_ann = display_df[mean_columns].round(3).astype(str).to_numpy()
    std_ann = display_df[std_columns].round(3).astype(str).to_numpy()

    fig_height = max(8.0, 2.8 + 0.65 * len(display_df))
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height), gridspec_kw={"width_ratios": [1.2, 0.9]})
    sns.heatmap(
        mean_norm,
        annot=mean_ann,
        fmt="",
        cmap=sns.color_palette("crest", as_cmap=True),
        linewidths=0.8,
        linecolor="#F0ECE4",
        cbar_kws={"label": "Column-wise normalized score"},
        xticklabels=["Mean IoU", "Weighted IoU", "Mean PSNR", "Mean SSIM"],
        yticklabels=display_df["row_label"],
        ax=axes[0],
    )
    axes[0].set_title("Central Tendency Summary")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        std_norm,
        annot=std_ann,
        fmt="",
        cmap=sns.color_palette("flare", as_cmap=True),
        linewidths=0.8,
        linecolor="#F0ECE4",
        cbar_kws={"label": "Column-wise normalized spread"},
        xticklabels=["Std IoU", "Std PSNR", "Std SSIM"],
        yticklabels=False,
        ax=axes[1],
    )
    axes[1].set_title("Frame-to-Frame Variability")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.suptitle("Metric Summary Heatmaps", fontsize=20, y=0.99)
    footer_lines = ["Cell colors are normalized within each metric column so different scales remain comparable."]
    if _contains_wild_sequence(summary_df):
        footer_lines.append(PROXY_IOU_NOTE)
    fig.text(0.5, 0.02, "\n".join(footer_lines), ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.90, bottom=0.18 if _contains_wild_sequence(summary_df) else 0.14, wspace=0.18)
    save_figure(fig, output_dir, stem)


def _blend_with_white(color, factor):
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(base * (1.0 - factor) + white * factor)


def style_table(table, cell_colors=None):
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDD6CA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#2F3E46")
            cell.set_text_props(color="white", weight="bold")
        else:
            default_bg = "#FFFDF9" if row % 2 else "#F5F0E8"
            if cell_colors and (row - 1, col) in cell_colors:
                cell.set_facecolor(cell_colors[(row - 1, col)])
            else:
                cell.set_facecolor(default_bg)


def build_gradient_cell_colors(dataframe, numeric_columns, cmap_name):
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    colors = {}
    start_index = len(dataframe.columns) - len(numeric_columns)
    for offset, column in enumerate(numeric_columns):
        series = dataframe[column].astype(float)
        min_value = series.min()
        max_value = series.max()
        for row_index, value in enumerate(series):
            if np.isclose(min_value, max_value):
                normalized = 0.5
            else:
                normalized = (value - min_value) / (max_value - min_value)
            colors[(row_index, start_index + offset)] = _blend_with_white(cmap(normalized), 0.20)
    return colors


def plot_summary_table(summary_df, output_dir, stem="11_metric_summary_table"):
    table_df = summary_df[
        [
            "sequence_label",
            "phase_label",
            "n_frames",
            "mean_iou",
            "weighted_iou",
            "mean_psnr",
            "mean_ssim",
            "evaluation_mode",
        ]
    ].copy()
    table_df.columns = ["Sequence", "Phase", "Frames", "Mean IoU", "Weighted IoU", "Mean PSNR", "Mean SSIM", "Mode"]
    table_df["Frames"] = table_df["Frames"].astype(int)
    for column in ["Mean IoU", "Weighted IoU", "Mean PSNR", "Mean SSIM"]:
        table_df[column] = table_df[column].map(lambda value: f"{value:.3f}")
    table_df = table_df[["Sequence", "Phase", "Frames", "Mean IoU", "Weighted IoU", "Mean PSNR", "Mean SSIM", "Mode"]]

    numeric_for_color = ["Mean IoU", "Weighted IoU", "Mean PSNR", "Mean SSIM"]
    color_df = table_df.copy()
    for column in numeric_for_color:
        color_df[column] = color_df[column].astype(float)
    cell_colors = build_gradient_cell_colors(color_df, numeric_for_color, "YlGnBu")

    fig_height = max(5.4, 1.8 + 0.48 * len(table_df))
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    style_table(table, cell_colors=cell_colors)
    ax.set_title("Metric Summary Table", fontsize=20, pad=18)
    if _contains_wild_sequence(summary_df):
        fig.text(0.5, 0.02, PROXY_IOU_NOTE, ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.88, bottom=0.12 if _contains_wild_sequence(summary_df) else 0.08)
    save_figure(fig, output_dir, stem)


def plot_paired_delta_summary_table(paired_summary_df, output_dir, stem="12_paired_delta_summary_table"):
    table_df = paired_summary_df[
        [
            "sequence_label",
            "pair_label",
            "mean_iou",
            "improved_iou",
            "worsened_iou",
            "mean_psnr",
            "mean_ssim",
        ]
    ].copy()
    table_df.columns = [
        "Sequence",
        "Pair",
        "Mean Delta IoU",
        "IoU Improved",
        "IoU Worsened",
        "Mean Delta PSNR",
        "Mean Delta SSIM",
    ]
    for column in ["Mean Delta IoU", "Mean Delta PSNR", "Mean Delta SSIM"]:
        table_df[column] = table_df[column].map(lambda value: f"{value:+.3f}")
    for column in ["IoU Improved", "IoU Worsened"]:
        table_df[column] = table_df[column].astype(int)

    color_df = table_df.copy()
    for column in ["Mean Delta IoU", "Mean Delta PSNR", "Mean Delta SSIM"]:
        color_df[column] = color_df[column].astype(float)
    cell_colors = build_gradient_cell_colors(color_df, ["Mean Delta IoU", "Mean Delta PSNR", "Mean Delta SSIM"], "RdYlGn")

    fig_height = max(4.8, 1.8 + 0.44 * len(table_df))
    fig, ax = plt.subplots(figsize=(17, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    style_table(table, cell_colors=cell_colors)
    ax.set_title("Paired Improvement Summary", fontsize=20, pad=18)
    if _contains_wild_sequence(paired_summary_df):
        fig.text(0.5, 0.02, PROXY_IOU_NOTE, ha="center", fontsize=11.5)
    finalize_figure_layout(fig, top=0.88, bottom=0.12 if _contains_wild_sequence(paired_summary_df) else 0.08)
    save_figure(fig, output_dir, stem)