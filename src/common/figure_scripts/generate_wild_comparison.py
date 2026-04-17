from pathlib import Path

try:
    from .metric_figure_output import ensure_output_dir
    from .metric_figure_plots import plot_wild_video_frames_comparison, set_plot_style
except ImportError:
    from metric_figure_output import ensure_output_dir
    from metric_figure_plots import plot_wild_video_frames_comparison, set_plot_style


def main():
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "results" / "visualizations" / "figures"
    ensure_output_dir(output_dir)
    set_plot_style()
    plot_wild_video_frames_comparison(output_dir)
    print(f"Saved figure to {output_dir / '13_wild_video_frames_comparison.png'}")

if __name__ == "__main__":
    main()
