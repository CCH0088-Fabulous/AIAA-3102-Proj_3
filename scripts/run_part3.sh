#!/usr/bin/env bash
set -euo pipefail

SEQUENCES=("bmx-trees" "tennis" "parkour" "dance-twirl" "wild_video_frames")

for SEQ in "${SEQUENCES[@]}"; do
  echo "======================================"
  echo "Running part 3 for sequence: $SEQ"
  echo "======================================"
  python src/part3_exploration/pipeline_part3.py \
    --common-config configs/common.yaml \
    --phase-config configs/part3_exploration.yaml \
    --sequence "$SEQ"
done
