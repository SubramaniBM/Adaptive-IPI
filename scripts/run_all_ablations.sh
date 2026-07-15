#!/bin/bash
set -e

echo "========================================================"
echo "Phase 5: Enron Ablation Study Training (Phases 5.1 - 5.3)"
echo "========================================================"

echo "--------------------------------------------------------"
echo "Training v2 (Enron +250)..."
python scripts/run_phase6_retrain.py --dataset-version 2

echo "--------------------------------------------------------"
echo "Training v3 (Enron +500)..."
python scripts/run_phase6_retrain.py --dataset-version 3

echo "--------------------------------------------------------"
echo "Training v4 (Enron +1000)..."
python scripts/run_phase6_retrain.py --dataset-version 4

echo "========================================================"
echo "Phase 6 & 7: Model Evaluation (Benchmark & OOD)"
echo "========================================================"
python scripts/run_enron_ablation_eval.py

echo "--------------------------------------------------------"
echo "Ablation Study Complete!"
