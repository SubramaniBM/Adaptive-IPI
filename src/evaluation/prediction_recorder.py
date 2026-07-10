"""
src/evaluation/prediction_recorder.py

Records raw predictions, failures, summaries, and failure clusters
for consumption by downstream adaptive phases.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd

from src.core.types import EvaluationResult
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionRecorder:
    """Records evaluation artifacts used for adaptive curriculum generation."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def record_all(self, result: EvaluationResult, predictions_df: pd.DataFrame) -> None:
        """Record all required artifacts for downstream phases."""
        self._record_summary(result)
        self._record_predictions(predictions_df)
        self._record_failures(predictions_df)
        logger.info(f"Prediction artifacts recorded to {self.output_dir}")

    def _record_summary(self, result: EvaluationResult) -> None:
        """Save machine-readable summary metrics."""
        summary = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "auroc": result.auroc,
            "auprc": result.auprc,
            "ece": result.ece,
            "tp": result.tp,
            "fp": result.fp,
            "tn": result.tn,
            "fn": result.fn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _record_predictions(self, df: pd.DataFrame) -> None:
        """Save raw predictions for all samples."""
        # Ensure we have a boolean 'correct' column
        if "correct" not in df.columns:
            df = df.copy()
            df["correct"] = df["predicted_label"] == df["label"]

        # The expected schema for predictions.csv
        expected_cols = [
            "id", "attack_family", "label", 
            "predicted_label", "correct", "predicted_prob", 
            "document_length", "attack_position"
        ]
        
        # Add optional columns if they exist
        if "teacher_entropy" in df.columns:
            expected_cols.append("teacher_entropy")

        # Map current df columns to expected schema where possible, filling missing with None
        out_df = pd.DataFrame(index=df.index)
        for col in expected_cols:
            if col in df.columns:
                out_df[col] = df[col]
            else:
                out_df[col] = None

        # Rename standard columns to match user spec exactly
        out_df = out_df.rename(columns={
            "id": "sample_id", 
            "label": "ground_truth",
            "predicted_label": "prediction",
            "predicted_prob": "probability"
        })

        out_df.to_csv(self.output_dir / "predictions.csv", index=False)

    def _record_failures(self, df: pd.DataFrame) -> None:
        """Save rich JSONL representations of incorrect predictions."""
        # Ensure 'correct' column
        if "correct" not in df.columns:
            df = df.copy()
            df["correct"] = df["predicted_label"] == df["label"]

        failures_df = df[~df["correct"]].copy()

        failures_list = []
        for _, row in failures_df.iterrows():
            failure = {
                "sample_id": row.get("id"),
                "attack_family": row.get("attack_family"),
                "ground_truth": int(row.get("label")),
                "prediction": int(row.get("predicted_label")),
                "probability": float(row.get("predicted_prob")),
                "email_text": row.get("email_text", row.get("retrieved_email")),
            }
            
            if "teacher_prediction" in row:
                failure["teacher_prediction"] = int(row["teacher_prediction"])
            if "teacher_entropy" in row:
                failure["teacher_entropy"] = float(row["teacher_entropy"])
                
            failures_list.append(failure)

        write_jsonl(failures_list, self.output_dir / "failures.jsonl")


