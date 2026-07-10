"""
src/teacher/parser.py

Parse and validate structured responses from the teacher model.

The teacher is expected to output JSON with prediction, confidence,
and reasoning fields. This module handles malformed responses
gracefully with fallback strategies and logging.
"""

import json
import math
import re
from typing import Optional

from src.core.types import TeacherAnnotation
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_teacher_response(
    sample_id: str,
    response_text: str,
) -> Optional[TeacherAnnotation]:
    """Parse a teacher model response into a TeacherAnnotation.

    Attempts to extract JSON from the response text. Handles cases
    where the model wraps JSON in markdown code blocks.

    Args:
        sample_id: The sample ID this response corresponds to.
        response_text: Raw text output from the teacher model.

    Returns:
        TeacherAnnotation if parsing succeeds, None if the response
        is malformed beyond recovery.
    """
    # Try to extract JSON from the response
    json_str = _extract_json(response_text)
    if json_str is None:
        logger.warning(
            f"  [Parser] Could not extract JSON from response for sample {sample_id}"
        )
        return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning(
            f"  [Parser] Invalid JSON for sample {sample_id}: {exc}"
        )
        return None

    return _build_annotation(sample_id, data)


def _extract_json(text: str) -> Optional[str]:
    """Extract a JSON object from text that may contain markdown formatting.

    Handles:
        - Bare JSON
        - JSON wrapped in ```json ... ``` blocks
        - JSON wrapped in ``` ... ``` blocks

    Args:
        text: Raw text potentially containing JSON.

    Returns:
        Extracted JSON string, or None.
    """
    # Try markdown code block first
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try finding a bare JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]

    return None


def _build_annotation(
    sample_id: str,
    data: dict,
) -> Optional[TeacherAnnotation]:
    """Build a TeacherAnnotation from parsed JSON data.

    Validates and normalises the parsed fields. Computes entropy
    from the probability distribution (Change #7).

    Args:
        sample_id: Sample identifier.
        data: Parsed JSON dictionary.

    Returns:
        TeacherAnnotation if valid, None otherwise.
    """
    # Extract prediction
    prediction = data.get("prediction")
    if prediction is None:
        logger.warning(f"  [Parser] Missing 'prediction' for sample {sample_id}")
        return None
    prediction = int(prediction)
    if prediction not in (0, 1):
        logger.warning(f"  [Parser] Invalid prediction value {prediction} for sample {sample_id}")
        return None

    # Extract confidence
    confidence = data.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Compute probability distribution
    if prediction == 1:
        probs = [1.0 - confidence, confidence]
    else:
        probs = [confidence, 1.0 - confidence]

    # Compute Shannon entropy (Change #7: store entropy for post-hoc analysis)
    entropy = _compute_entropy(probs)

    # Extract reasoning
    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return TeacherAnnotation(
        id=sample_id,
        teacher_prediction=prediction,
        teacher_probs=probs,
        teacher_entropy=entropy,
        teacher_reasoning=reasoning,
    )


def _compute_entropy(probs: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        probs: List of probabilities (should sum to 1.0).

    Returns:
        Shannon entropy in bits.
    """
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
