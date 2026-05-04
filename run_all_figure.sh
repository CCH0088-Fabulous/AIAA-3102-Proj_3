#!/usr/bin/env bash
set -euo pipefail

python src/common/figure_scripts/generate_metric_figures.py \
  --metrics-root results/metrics \
  --output-dir results/visualizations/figures

python src/common/figure_scripts/generate_metric_figures_davis.py \
  --metrics-root results/metrics \
  --output-dir results/visualizations/figures/davis

python src/common/figure_scripts/generate_wild_comparison.py