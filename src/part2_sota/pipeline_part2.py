import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from common.config import ensure_phase_output_dirs, load_yaml_config, resolve_sequence_spec


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2 SOTA pipeline entrypoint stub.")
    parser.add_argument("--sequence", default=None, help="Sequence key or direct frame folder.")
    parser.add_argument("--common-config", default="configs/common.yaml")
    parser.add_argument("--phase-config", default="configs/part2_sota.yaml")
    return parser.parse_args()


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

    segmentation_backend = phase_cfg.get("models", {}).get("segmentation", {}).get("backend")
    inpainting_backend = phase_cfg.get("models", {}).get("inpainting", {}).get("backend")

    print("Phase 2 interface alignment complete.")
    print(f"Resolved sequence: {sequence_spec['output_name']}")
    print(f"Segmentation backend: {segmentation_backend}")
    print(f"Inpainting backend: {inpainting_backend}")
    print("Implementation of the SOTA pipeline is scheduled for Phase 4.")


if __name__ == "__main__":
    main()