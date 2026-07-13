#!/bin/bash
set -e

echo "Starting Experiment 2 (KD)..."
source .venv/bin/activate
python scripts/run_phase3_distill.py --loss-type kd --dataset-version 1
echo "Experiment 2 finished."

echo "Starting Experiment 3 (Confidence Weighting)..."
python scripts/run_phase3_distill.py --loss-type cw --dataset-version 1
echo "Experiment 3 finished."

echo "Starting Experiment 4 (Temperature Scaling)..."
python scripts/run_exp4_temperature_scaling.py
echo "Experiment 4 finished."

echo "All overnight experiments completed successfully."
