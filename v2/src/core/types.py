"""
v2/src/core/types.py

Data structures and types for span-level detection and sanitization.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class InjectionSpan:
    start_char: int
    end_char: int
    text: str
    confidence: float


@dataclass
class SanitizationResult:
    original_text: str
    sanitized_text: str
    has_injection: bool
    spans: List[InjectionSpan]
    redacted_text: str
