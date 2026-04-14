#!/bin/bash
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part1_baseline.yaml --sequence tennis
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part1_baseline.yaml --sequence bmx-trees
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part2_sota.yaml --sequence tennis
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part2_sota.yaml --sequence bmx-trees
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part3_exploration.yaml --sequence tennis
conda run -n project3 python scripts/evaluate_metrics.py --phase-config configs/part3_exploration.yaml --sequence bmx-trees
