"""
scripts/run_phase5_generate.py

Phase 5: Adaptive Hard-Negative Generation
───────────────────────────────────────────
Select student failures, generate targeted hard-negative examples
using the teacher, verify them, and merge into a new dataset version.

NOTE: This phase requires implementing the research methodology in:
    - src/adaptive/failure_selection.py
    - src/adaptive/hard_negative_generation.py

Until those are implemented, this script will raise NotImplementedError.

Usage:
    python scripts/run_phase5_generate.py --failure-report experiments/exp002/failure_report.json
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Adaptive Hard-Negative Generation"
    )
    parser.add_argument(
        "--failure-report", type=Path, required=True,
        help="Path to failure report JSON from Phase 4.",
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--teacher-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "teacher.yaml",
    )
    parser.add_argument(
        "--adaptive-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "adaptive.yaml",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
        help="Source dataset version to augment.",
    )
    args = parser.parse_args()

    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase5.log")

    logger.info("━" * 60)
    logger.info("PHASE 5: Adaptive Hard-Negative Generation")
    logger.info("━" * 60)

    # This phase requires implementing the research methodology.
    # The pipeline structure is ready; the methodology must be filled in.
    logger.error(
        "Phase 5 requires implementing the research methodology.\n"
        "  1. Implement failure selection:  src/adaptive/failure_selection.py\n"
        "  2. Implement generation:         src/adaptive/hard_negative_generation.py\n"
        "  3. Update config:                configs/adaptive.yaml"
    )
    raise NotImplementedError(
        "Phase 5 methodology not yet implemented. See the log message above."
    )


if __name__ == "__main__":
    main()
