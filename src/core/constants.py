"""
src/core/constants.py

Project-wide constants for the Adaptive-IPI project.

All fixed values (seeds, model identifiers, column names, etc.) are
defined here. Configuration values that may vary between experiments
belong in YAML configs, not here.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED: int = 42

# ── Model identifiers ────────────────────────────────────────────────────────
TEACHER_MODEL_ID: str = "Qwen/Qwen3-32B-Instruct"
STUDENT_MODEL_ID: str = "answerdotai/ModernBERT-base"

# ── Number of classes ─────────────────────────────────────────────────────────
NUM_LABELS: int = 2

# ── Unified dataset schema ────────────────────────────────────────────────────
# Column names used in all processed DataFrames and CSV files.
SCHEMA_COLUMNS: list[str] = [
    "id",
    "context",
    "attack_instruction",
    "label",
    "attack_family",
    "attack_position",
    "split",
    "task",
    "source",
]

# ── Teacher annotation schema ────────────────────────────────────────────────
# Fields stored in teacher_annotations.jsonl (Change #7: rich annotations).
TEACHER_ANNOTATION_COLUMNS: list[str] = [
    "id",
    "teacher_prediction",
    "teacher_probs",
    "teacher_entropy",
    "teacher_reasoning",
]

# ── Dataset source identifier ────────────────────────────────────────────────
SOURCE_BIPIA: str = "bipia"
SOURCE_GENERATED: str = "generated"

# ── BIPIA task names ──────────────────────────────────────────────────────────
BIPIA_TASKS: list[str] = ["email", "qa", "abstract", "table", "code"]

# ── Dataset versioning prefix (Change #4) ─────────────────────────────────────
DATASET_VERSION_PREFIX: str = "dataset_v"
