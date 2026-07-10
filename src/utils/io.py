"""
src/utils/io.py

File I/O utilities for the Adaptive-IPI project.

Provides helpers for reading/writing JSONL, CSV, and general-purpose
file operations used across the pipeline.
"""

import csv
import json
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── JSONL ─────────────────────────────────────────────────────────────────────


def read_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"Malformed JSON at {path}:{line_no} — {exc}")
    return records


def iter_jsonl(path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    """Lazily iterate over records in a JSONL file.

    Useful for large files that should not be loaded entirely into memory.

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed JSON records.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(
    records: list[dict[str, Any]],
    path: Union[str, Path],
    append: bool = False,
) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        records: List of records to write.
        path: Output file path.
        append: If True, append to existing file instead of overwriting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: Union[str, Path]) -> None:
    """Append a single record to a JSONL file.

    Creates the file and parent directories if they don't exist.

    Args:
        record: Single record to append.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── CSV ───────────────────────────────────────────────────────────────────────


def read_csv(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments passed to ``pd.read_csv``.

    Returns:
        DataFrame with the file contents.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs: Any) -> None:
    """Write a DataFrame to a CSV file.

    Args:
        df: DataFrame to write.
        path: Output file path.
        **kwargs: Additional arguments passed to ``df.to_csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


# ── JSON ──────────────────────────────────────────────────────────────────────


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Write data to a JSON file.

    Args:
        data: Data to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ── Directory helpers ─────────────────────────────────────────────────────────


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
