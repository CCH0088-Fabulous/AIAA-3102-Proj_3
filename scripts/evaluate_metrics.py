import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.config import ensure_phase_output_dirs, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description="Unified metrics entrypoint stub.")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument(
        "--phase-config",
        default="configs/part1_baseline.yaml",
        help="Phase config whose outputs should be evaluated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    common_cfg = load_yaml_config(args.common_config)
    phase_cfg = load_yaml_config(args.phase_config)
    ensure_phase_output_dirs(phase_cfg)

    metrics_root = common_cfg.get("paths", {}).get("metrics_root", "results/metrics")
    print("Metrics interface alignment complete.")
    print(f"Metrics root: {metrics_root}")
    print("Metric computation will be implemented in Phase 3.")


if __name__ == "__main__":
    main()