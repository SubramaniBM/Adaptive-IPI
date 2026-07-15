#!/bin/bash
set -e

echo "========================================================"
echo "Running Seed 43 on Enron1000 (dataset_v4)"
echo "========================================================"
python scripts/run_phase6_retrain.py --dataset-version 4 --seed 43

echo "========================================================"
echo "Running Seed 44 on Enron1000 (dataset_v4)"
echo "========================================================"
python scripts/run_phase6_retrain.py --dataset-version 4 --seed 44

echo "========================================================"
echo "Multi-seed training complete! Running Evaluation..."
echo "========================================================"
python scripts/run_enron_ablation_eval.py
