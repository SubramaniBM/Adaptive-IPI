"""
scripts/recover_annotations.py

Offline recovery utility that attempts to repair malformed JSON responses
from `annotation_errors.csv` using a deterministic repair library.

Only appends to `teacher_annotations.jsonl` if it passes strict schema validation.
Unrecoverable samples are left in the errors CSV.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import json_repair

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.teacher.parser import _build_annotation, _extract_json
from src.utils.io import append_jsonl

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Recover malformed teacher annotations.")
    parser.add_argument(
        "--errors-csv", type=Path, required=True,
        help="Path to annotation_errors.csv",
    )
    parser.add_argument(
        "--output-jsonl", type=Path, required=True,
        help="Path to teacher_annotations.jsonl to append to",
    )
    args = parser.parse_args()

    if not args.errors_csv.exists():
        logger.info(f"No error CSV found at {args.errors_csv}. Nothing to recover.")
        return

    # Read all errors
    unrecoverable_rows = []
    recovered_count = 0
    total_count = 0

    with open(args.errors_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            total_count += 1
            sample_id = row["sample_id"]
            raw_response = row["raw_response"]
            
            # Try to extract JSON block first
            try:
                json_str = _extract_json(raw_response)
                if json_str is None:
                    json_str = raw_response
            except:
                json_str = raw_response
                
            # Attempt repair
            try:
                repaired_data = json_repair.repair_json(json_str, return_objects=True)
                
                if not isinstance(repaired_data, dict):
                    raise ValueError("Repaired output is not a dictionary.")
                    
                # Strict schema validation using existing parser logic
                annotation = _build_annotation(sample_id, repaired_data)
                
                # Append to annotations
                import dataclasses
                append_jsonl(dataclasses.asdict(annotation), args.output_jsonl)
                recovered_count += 1
                logger.info(f"  [+] Recovered: {sample_id}")
                
            except Exception as e:
                # Still unrecoverable
                unrecoverable_rows.append(row)
                logger.debug(f"  [-] Unrecoverable {sample_id}: {e}")

    # Rewrite the CSV with only unrecoverable
    if unrecoverable_rows:
        with open(args.errors_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(unrecoverable_rows)
    else:
        # All recovered, can delete or empty the file
        args.errors_csv.unlink()

    logger.info("━" * 50)
    logger.info("Recovery Complete")
    logger.info(f"  Total errors analyzed: {total_count}")
    logger.info(f"  Successfully recovered: {recovered_count}")
    logger.info(f"  Still unrecoverable: {len(unrecoverable_rows)}")
    logger.info("━" * 50)


if __name__ == "__main__":
    main()
