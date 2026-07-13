#!/bin/bash
set -e

echo "============================================================"
echo " Starting Adaptive-IPI FULL RUN (dataset_v0 -> v1)"
echo "============================================================"

# Ensure PYTHONPATH is set
export PYTHONPATH="."

# Run Phase 2: Teacher Annotation (using Ollama)
echo ""
echo "▶ Running Phase 2: Teacher Annotation (Estimated Time: ~2 hours)..."
python scripts/run_phase2_teacher.py \
    --config configs/mac_teacher.yaml \
    --data-config configs/mac_data.yaml \
    --dataset-version 0

# Run Phase 3: Initial Distillation (using MPS)
echo ""
echo "▶ Running Phase 3: Initial Distillation (Epochs: 3)..."
python scripts/run_phase3_distill.py \
    --student-config configs/mac_student.yaml \
    --data-config configs/mac_data.yaml \
    --dataset-version 0

# Run Phase 4: Evaluation
echo ""
echo "▶ Running Phase 4: Failure Analysis..."
LATEST_EXP=$(ls -td outputs/mac_experiments/exp* | head -1)
CHECKPOINT_DIR="${LATEST_EXP}/checkpoint/best"

python scripts/run_phase4_analyze.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --split validation \
    --data-config configs/mac_data.yaml \
    --dataset-version 0

# Grab the failure report that was just generated in the experiments/ folder
# (Since Phase 4 saves to experiments/ instead of outputs/mac_experiments by default)
LATEST_REPORT_EXP=$(ls -td experiments/exp* | head -1)
REPORT_PATH="${LATEST_REPORT_EXP}/failure_report.json"

echo ""
echo "▶ Found failure report at: ${REPORT_PATH}"

# Run Phase 5: Curriculum Generation
echo ""
echo "▶ Running Phase 5: Curriculum Generation..."
python scripts/run_phase5_generate.py \
    --failure-report "${REPORT_PATH}" \
    --teacher-config configs/mac_teacher.yaml \
    --data-config configs/mac_data.yaml \
    --dataset-version 0

echo ""
echo "============================================================"
echo " Halfway point: Curriculum generated (dataset_v1)!"
echo "============================================================"

# Run Phase 6: Retraining on Augmented Dataset
echo ""
echo "▶ Running Phase 6: Retraining Student on Augmented Curriculum..."
python scripts/run_phase3_distill.py \
    --student-config configs/mac_student.yaml \
    --data-config configs/mac_data.yaml \
    --dataset-version 1

# Run Phase 7: Final Evaluation
echo ""
echo "▶ Running Phase 7: Final Evaluation..."
NEW_EXP=$(ls -td outputs/mac_experiments/exp* | head -1)
NEW_CHECKPOINT="${NEW_EXP}/checkpoint/best"

python scripts/run_phase4_analyze.py \
    --checkpoint "${NEW_CHECKPOINT}" \
    --split validation \
    --data-config configs/mac_data.yaml \
    --dataset-version 1

echo ""
echo "============================================================"
echo " Adaptive-IPI Mac Full Dataset Loop 100% Completed!"
echo "============================================================"
