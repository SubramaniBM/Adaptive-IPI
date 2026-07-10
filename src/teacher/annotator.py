"""
src/teacher/annotator.py

Teacher annotation orchestration for the Adaptive-IPI project.

Coordinates the teacher annotation pipeline: iterates over samples,
sends them to the inference backend, parses responses, and saves
results with checkpointing support for resumable runs.

Stores rich annotations per Change #7:
    id, teacher_prediction, teacher_probs, teacher_entropy, teacher_reasoning
"""

import dataclasses
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.core.types import TeacherAnnotation
from src.teacher.backend import BaseTeacherBackend, create_backend
from src.teacher.parser import parse_teacher_response
from src.teacher.prompts import build_messages
from src.utils.io import append_jsonl, read_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TeacherAnnotator:
    """Orchestrates teacher model annotation of dataset samples.

    Features:
        - Configurable inference backend (vLLM / Transformers / API)
        - Checkpointing: saves progress incrementally, can resume on failure
        - Rich annotations: probs, entropy, reasoning (Change #7)
        - Batch processing support

    Attributes:
        backend: Teacher inference backend.
        output_path: Path to the output JSONL file.
        batch_size: Number of samples to process per batch.
    """

    def __init__(
        self,
        backend: BaseTeacherBackend,
        output_path: Union[str, Path],
        batch_size: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialise the annotator.

        Args:
            backend: Teacher inference backend instance.
            output_path: Path to save annotations (JSONL format).
            batch_size: Number of samples per batch for batched backends.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate per response.
            system_prompt: Optional override for the system prompt.
        """
        self.backend = backend
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def annotate(
        self,
        df: pd.DataFrame,
        resume: bool = True,
    ) -> list[TeacherAnnotation]:
        """Annotate all samples in a DataFrame.

        Args:
            df: DataFrame with at least 'id' and 'text' columns.
            resume: If True, skip samples that already have annotations
                in the output file.

        Returns:
            List of all TeacherAnnotation objects (including resumed ones).
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing annotations for resume
        completed_ids: set[str] = set()
        existing_annotations: list[TeacherAnnotation] = []

        if resume and self.output_path.exists():
            existing_records = read_jsonl(self.output_path)
            for record in existing_records:
                completed_ids.add(record["id"])
                existing_annotations.append(
                    TeacherAnnotation(**{
                        k: record[k] for k in TeacherAnnotation.__dataclass_fields__
                    })
                )
            logger.info(f"Resuming: {len(completed_ids):,} samples already annotated")

        # Filter to unannotated samples
        pending_df = df[~df["id"].isin(completed_ids)]
        logger.info(f"Annotating {len(pending_df):,} samples (batch_size={self.batch_size})")

        if len(pending_df) == 0:
            logger.info("All samples already annotated")
            return existing_annotations

        # Process samples
        new_annotations: list[TeacherAnnotation] = []
        failed_count = 0

        if self.batch_size > 1:
            new_annotations, failed_count = self._annotate_batched(pending_df)
        else:
            new_annotations, failed_count = self._annotate_sequential(pending_df)

        all_annotations = existing_annotations + new_annotations
        logger.info(
            f"Annotation complete: {len(new_annotations):,} new, "
            f"{failed_count:,} failed, {len(all_annotations):,} total"
        )

        return all_annotations

    def _annotate_sequential(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[TeacherAnnotation], int]:
        """Annotate samples one at a time with per-sample checkpointing."""
        annotations: list[TeacherAnnotation] = []
        failed_count = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Teacher annotation"):
            sample_id = row["id"]
            text = row["text"]

            try:
                messages = build_messages(text, system_prompt=self.system_prompt)
                response = self.backend.generate(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                annotation = parse_teacher_response(sample_id, response)

                if annotation is not None:
                    annotations.append(annotation)
                    # Checkpoint: append immediately
                    append_jsonl(dataclasses.asdict(annotation), self.output_path)
                else:
                    failed_count += 1
                    logger.warning(f"Failed to parse response for sample {sample_id}")

            except Exception as exc:
                failed_count += 1
                logger.error(f"Error annotating sample {sample_id}: {exc}")

        return annotations, failed_count

    def _annotate_batched(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[TeacherAnnotation], int]:
        """Annotate samples in batches for backends with native batching."""
        annotations: list[TeacherAnnotation] = []
        failed_count = 0

        # Build batches
        rows = list(df.iterrows())
        batches = [
            rows[i : i + self.batch_size]
            for i in range(0, len(rows), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Teacher annotation (batched)"):
            batch_ids = [row["id"] for _, row in batch]
            batch_messages = [
                build_messages(row["text"], system_prompt=self.system_prompt)
                for _, row in batch
            ]

            try:
                responses = self.backend.generate_batch(
                    batch_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                batch_records: list[dict] = []
                for sample_id, response in zip(batch_ids, responses):
                    annotation = parse_teacher_response(sample_id, response)
                    if annotation is not None:
                        annotations.append(annotation)
                        batch_records.append(dataclasses.asdict(annotation))
                    else:
                        failed_count += 1

                # Checkpoint: write batch
                if batch_records:
                    write_jsonl(batch_records, self.output_path, append=True)

            except Exception as exc:
                failed_count += len(batch)
                logger.error(f"Error in batch annotation: {exc}")

        return annotations, failed_count


def create_annotator(
    config: dict[str, Any],
    output_path: Union[str, Path],
) -> TeacherAnnotator:
    """Factory function to create a TeacherAnnotator from config.

    Args:
        config: Teacher configuration dictionary (from teacher.yaml).
        output_path: Path to save annotations.

    Returns:
        Configured TeacherAnnotator instance.
    """
    backend = create_backend(
        backend_type=config["backend"],
        model_id=config["model_id"],
        **config.get("backend_kwargs", {}),
    )

    return TeacherAnnotator(
        backend=backend,
        output_path=output_path,
        batch_size=config.get("batch_size", 1),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 1024),
        system_prompt=config.get("system_prompt"),
    )
