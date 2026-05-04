#!/usr/bin/env bash
set -euo pipefail

# 默认在 common.yaml 里配置了五个序列
SEQUENCES=("bmx-trees" "tennis" "parkour" "dance-twirl" "wild_video_frames")

for SEQ in "${SEQUENCES[@]}"; do
  echo "======================================"
  echo "Running part 1 for sequence: $SEQ"
  echo "======================================"
  python src/part1_baseline/pipeline_part1.py \
    --common-config configs/common.yaml \
    --phase-config configs/part1_baseline.yaml \
    --sequence "$SEQ"
done
