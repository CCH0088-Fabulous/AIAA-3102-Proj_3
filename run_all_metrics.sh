#!/usr/bin/env bash
set -euo pipefail

PHASE_CONFIGS=(
	"configs/part1_baseline.yaml"
	"configs/part2_sota.yaml"
	"configs/part3_exploration.yaml"
)

SEQUENCES=("bmx-trees" "tennis" "parkour" "dance-twirl" "wild_video_frames")

for PHASE_CONFIG in "${PHASE_CONFIGS[@]}"; do
	for SEQ in "${SEQUENCES[@]}"; do
		echo "======================================"
		echo "Evaluating phase config: $PHASE_CONFIG"
		echo "Sequence: $SEQ"
		echo "======================================"
		python scripts/evaluate_metrics.py \
			--phase-config "$PHASE_CONFIG" \
			--sequence "$SEQ"
	done
done
