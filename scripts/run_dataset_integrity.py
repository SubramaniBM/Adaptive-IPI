"""
scripts/run_dataset_integrity.py

Task 1: Dataset Integrity Audit
────────────────────────────────
Verifies the frozen dataset meets all integrity constraints.

Outputs: reports/dataset_integrity_report.json
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io import read_csv
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    setup_logging()

    dataset_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    reports_dir = _PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_csv(dataset_dir / "train.csv")
    val_df = read_csv(dataset_dir / "validation.csv")
    test_df = read_csv(dataset_dir / "test.csv")

    import pandas as pd
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    report = {
        "dataset_version": "dataset_v0",
        "total_samples": len(all_df),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "checks": {},
        "passed": True,
    }

    # ── Check 1: Zero overlap of email contexts between splits ────────
    train_contexts = set(train_df["context"].unique())
    val_contexts = set(val_df["context"].unique())
    test_contexts = set(test_df["context"].unique())

    train_val_overlap = train_contexts & val_contexts
    train_test_overlap = train_contexts & test_contexts
    val_test_overlap = val_contexts & test_contexts

    context_check = {
        "name": "zero_context_overlap",
        "train_unique_contexts": len(train_contexts),
        "validation_unique_contexts": len(val_contexts),
        "test_unique_contexts": len(test_contexts),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "passed": len(train_val_overlap) == 0
                  and len(train_test_overlap) == 0
                  and len(val_test_overlap) == 0,
    }
    report["checks"]["context_overlap"] = context_check
    if not context_check["passed"]:
        report["passed"] = False

    # ── Check 2: Zero duplicate IDs ───────────────────────────────────
    dup_ids = all_df["id"].duplicated().sum()
    id_check = {
        "name": "zero_duplicate_ids",
        "duplicate_count": int(dup_ids),
        "passed": dup_ids == 0,
    }
    report["checks"]["duplicate_ids"] = id_check
    if not id_check["passed"]:
        report["passed"] = False

    # ── Check 3: Zero duplicate (context, user_intent) pairs ──────────
    dup_pairs = all_df.duplicated(subset=["context", "attack_instruction"]).sum()
    pair_check = {
        "name": "zero_duplicate_context_intent_pairs",
        "duplicate_count": int(dup_pairs),
        "passed": dup_pairs == 0,
    }
    report["checks"]["duplicate_pairs"] = pair_check
    if not pair_check["passed"]:
        report["passed"] = False

    # ── Check 4: Labels are binary ────────────────────────────────────
    invalid_labels = (~all_df["label"].isin([0, 1])).sum()
    label_check = {
        "name": "binary_labels",
        "invalid_count": int(invalid_labels),
        "unique_labels": sorted(all_df["label"].unique().tolist()),
        "passed": invalid_labels == 0,
    }
    report["checks"]["binary_labels"] = label_check
    if not label_check["passed"]:
        report["passed"] = False

    # ── Check 5: Attack family coverage ───────────────────────────────
    attacks = all_df[all_df["label"] == 1]
    unique_families = sorted(attacks["attack_family"].unique().tolist())
    family_check = {
        "name": "attack_family_coverage",
        "unique_families": len(unique_families),
        "families": unique_families,
        "passed": len(unique_families) >= 15,
    }
    report["checks"]["attack_family_coverage"] = family_check
    if not family_check["passed"]:
        report["passed"] = False

    # ── Check 6: Attack position coverage ─────────────────────────────
    actual_positions = set(attacks["attack_position"].unique())
    expected_positions = {"start", "middle", "end"}
    position_check = {
        "name": "attack_position_coverage",
        "positions_found": sorted(list(actual_positions)),
        "positions_expected": sorted(list(expected_positions)),
        "passed": actual_positions == expected_positions,
    }
    report["checks"]["attack_position_coverage"] = position_check
    if not position_check["passed"]:
        report["passed"] = False

    # ── Save report ───────────────────────────────────────────────────
    # Convert numpy types for JSON serialization
    import numpy as np
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    report = convert(report)

    output_path = reports_dir / "dataset_integrity_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if report["passed"]:
        logger.info("✅ All dataset integrity checks PASSED.")
    else:
        logger.error("❌ Some dataset integrity checks FAILED. See report.")

    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
