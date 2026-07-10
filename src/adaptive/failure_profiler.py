"""
src/adaptive/failure_profiler.py

Computes deterministic failure statistics from evaluation outputs.
No machine learning or clustering is used here.
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any

import pandas as pd

from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FailureProfiler:
    """Profiles evaluation failures deterministically and extracts representative samples."""

    def __init__(self, output_dir: Union[str, Path], max_representative: int = 20):
        self.output_dir = Path(output_dir)
        self.max_representative = max_representative
        ensure_dir(self.output_dir)

    def generate_profile(self, predictions_csv: Union[str, Path], failures_jsonl: Union[str, Path]) -> None:
        """Generate failure_profile.json and representative_failures.jsonl."""
        pred_df = pd.read_csv(predictions_csv)
        
        # Load failures directly for rich text context
        failures = []
        with open(failures_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                failures.append(json.loads(line))
        
        profile = self._compute_statistics(pred_df)
        
        # Extract representative failures
        rep_failures = self._select_representative_failures(pred_df, failures)
        profile["representative_failure_ids"] = [f["sample_id"] for f in rep_failures]
        
        # Save artifacts
        with open(self.output_dir / "failure_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
            
        write_jsonl(rep_failures, self.output_dir / "representative_failures.jsonl")
        logger.info(f"Generated failure profile and {len(rep_failures)} representative failures at {self.output_dir}")

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute deterministic statistics over the predictions."""
        if "correct" not in df.columns:
            df["correct"] = df["prediction"] == df["ground_truth"]

        failures_df = df[~df["correct"]]
        correct_df = df[df["correct"]]
        
        # Low confidence correct (probability close to 0.5, e.g., margin < 0.2)
        margin = (df["probability"] - 0.5).abs()
        df_margin = df.copy()
        df_margin["margin"] = margin
        
        low_conf_correct = df_margin[(df_margin["correct"]) & (df_margin["margin"] < 0.2)]

        stats: Dict[str, Any] = {
            "total_samples": int(len(df)),
            "total_failures": int(len(failures_df)),
            "false_positives": int(len(failures_df[failures_df["prediction"] == 1])),
            "false_negatives": int(len(failures_df[failures_df["prediction"] == 0])),
            "low_confidence_correct": int(len(low_conf_correct)),
            "attack_position_distribution": {},
            "document_length_distribution": {},
            "attack_family_distribution": {},
            "confidence_statistics": {}
        }
        
        if len(failures_df) == 0:
            return stats

        # Attack Position Distribution
        if "attack_position" in failures_df.columns:
            pos_counts = failures_df["attack_position"].value_counts(dropna=False)
            stats["attack_position_distribution"] = {str(k): int(v) for k, v in pos_counts.items()}

        # Document Length Distribution
        if "document_length" in failures_df.columns:
            long_docs = failures_df[failures_df["document_length"] > 500]
            short_docs = failures_df[failures_df["document_length"] <= 500]
            stats["document_length_distribution"] = {
                "long_documents_gt_500": int(len(long_docs)),
                "short_documents_lte_500": int(len(short_docs)),
            }

        # Attack Family Distribution
        if "attack_family" in failures_df.columns:
            fam_counts = failures_df["attack_family"].value_counts(dropna=False)
            stats["attack_family_distribution"] = {str(k): int(v) for k, v in fam_counts.items()}

        # Confidence statistics
        probs = failures_df["probability"]
        stats["confidence_statistics"] = {
            "mean_probability": float(probs.mean()),
            "min_probability": float(probs.min()),
            "max_probability": float(probs.max()),
        }

        return stats

    def _select_representative_failures(self, df: pd.DataFrame, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select top N failures ranked by lowest confidence margin."""
        if not failures:
            return []
            
        failures_df = df[~df["correct"]].copy()
        if len(failures_df) == 0:
            return []
            
        # Margin is distance from 0.5 (decision boundary)
        failures_df["margin"] = (failures_df["probability"] - 0.5).abs()
        
        # Sort by lowest margin (most uncertain)
        sorted_failures = failures_df.sort_values("margin", ascending=True)
        top_ids = set(sorted_failures.head(self.max_representative)["sample_id"].tolist())
        
        # Filter and order original failure records to match top_ids
        rep_failures = []
        for sample_id in sorted_failures.head(self.max_representative)["sample_id"]:
            # Find the dict in failures
            match = next((f for f in failures if f.get("sample_id") == sample_id), None)
            if match:
                rep_failures.append(match)
                
        return rep_failures
