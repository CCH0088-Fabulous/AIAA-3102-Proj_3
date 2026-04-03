#!/usr/bin/env bash
set -euo pipefail

python src/part3_exploration/pipeline_part3.py \
  --common-config configs/common.yaml \
  --phase-config configs/part3_exploration.yaml \
  "$@"