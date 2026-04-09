import argparse
from pathlib import Path

try:
    from .metric_figure_data import compute_paired_deltas, compute_summaries, load_metric_data
    from .metric_figure_output import ensure_output_dir, remove_stale_pdf_figures, write_manifest, write_summary_csvs
    from .metric_figure_plots import (
        plot_iou_distribution,
        plot_iou_ecdf,
        plot_iou_vs_union,
        plot_paired_delta_summary_table,
        plot_paired_deltas,
        plot_quality_distribution,
        plot_quality_vs_valid_pixels,
        plot_sequence_trends,
        plot_summary_heatmaps,
        plot_summary_table,
        set_plot_style,
    )
except ImportError:
    from metric_figure_data import compute_paired_deltas, compute_summaries, load_metric_data
    from metric_figure_output import ensure_output_dir, remove_stale_pdf_figures, write_manifest, write_summary_csvs
    from metric_figure_plots import (
        plot_iou_distribution,
        plot_iou_ecdf,
        plot_iou_vs_union,
        plot_paired_delta_summary_table,
        plot_paired_deltas,
        plot_quality_distribution,
        plot_quality_vs_valid_pixels,
        plot_sequence_trends,
        plot_summary_heatmaps,
        plot_summary_table,
        set_plot_style,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-quality metric figures.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--metrics-root",
        default=str(repo_root / "results" / "metrics"),
        help="Root directory containing phase/sequence metric CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "results" / "visualizations" / "figures"),
        help="Directory where figures and summary CSVs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_root = Path(args.metrics_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)
    remove_stale_pdf_figures(output_dir)
    set_plot_style()

    iou_df, quality_df = load_metric_data(metrics_root)
    summary_df = compute_summaries(iou_df, quality_df)
    deltas_df, paired_summary_df = compute_paired_deltas(iou_df, quality_df)

    plot_iou_distribution(iou_df, summary_df, output_dir)
    plot_quality_distribution(quality_df, summary_df, output_dir)

    for sequence in iou_df["sequence"].cat.categories:
        plot_sequence_trends(iou_df, quality_df, output_dir, sequence)
        plot_paired_deltas(deltas_df, output_dir, sequence)

    plot_iou_ecdf(iou_df, output_dir)
    plot_iou_vs_union(iou_df, output_dir)
    plot_quality_vs_valid_pixels(quality_df, output_dir)
    plot_summary_heatmaps(summary_df, output_dir)
    plot_summary_table(summary_df, output_dir)
    plot_paired_delta_summary_table(paired_summary_df, output_dir)

    write_summary_csvs(summary_df, paired_summary_df, output_dir)
    write_manifest(output_dir)

    print(f"Saved figures and summary tables to {output_dir}")


if __name__ == "__main__":
    main()