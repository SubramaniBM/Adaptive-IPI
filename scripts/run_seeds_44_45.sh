#!/bin/bash
set -e

source .venv/bin/activate

# SEED 44
echo "======================================"
echo "RUNNING SEED 44"
echo "======================================"
python scripts/run_phase3_distill.py --student-config configs/mac_student.yaml --data-config configs/mac_data_seed44.yaml --dataset-version 1 --loss-type kd --force
EXP44=$(ls -td outputs/mac_experiments/exp* | head -1)
echo "Seed 44 trained in $EXP44"
python scripts/run_phase7_evaluate.py --checkpoint "$EXP44/checkpoint/best" --dataset-version 1

# SEED 45
echo "======================================"
echo "RUNNING SEED 45"
echo "======================================"
python scripts/run_phase3_distill.py --student-config configs/mac_student.yaml --data-config configs/mac_data_seed45.yaml --dataset-version 1 --loss-type kd --force
EXP45=$(ls -td outputs/mac_experiments/exp* | head -1)
echo "Seed 45 trained in $EXP45"
python scripts/run_phase7_evaluate.py --checkpoint "$EXP45/checkpoint/best" --dataset-version 1

echo "======================================"
echo "ALL REPRODUCIBILITY RUNS COMPLETE"
echo "======================================"
