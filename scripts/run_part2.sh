#!/usr/bin/env bash
set -euo pipefail

python src/part2_sota/pipeline_part2.py \
  --common-config configs/common.yaml \
  --phase-config configs/part2_sota.yaml \
  "$@"