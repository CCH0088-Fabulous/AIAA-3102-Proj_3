#!/usr/bin/env bash
set -euo pipefail

SEQUENCES=("bmx-trees" "tennis" "parkour" "dance-twirl" "wild_video_frames")

for SEQ in "${SEQUENCES[@]}"; do
  echo "======================================"
  echo "Running part 2 for sequence: $SEQ"
  echo "======================================"
  python src/part2_sota/pipeline_part2.py \
    --common-config configs/common.yaml \
    --phase-config configs/part2_sota.yaml \
    --sequence "$SEQ"
done
