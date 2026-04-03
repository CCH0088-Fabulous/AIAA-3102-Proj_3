#!/usr/bin/env bash
set -euo pipefail

python src/part1_baseline/pipeline_part1.py \
  --common-config configs/common.yaml \
  --phase-config configs/part1_baseline.yaml \
  "$@"