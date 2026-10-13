"""
v2/src/inference/sanitizer.py

Inference pipeline for span extraction, threat localization, and document sanitization.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, PreTrainedModel

from v2.src.core.types import InjectionSpan, SanitizationResult


class IPISanitizer:
    """
    Production-grade inference pipeline that detects injection spans
    and sanitizes the document to allow safe downstream LLM consumption.
    """

    LABEL_O = 0
    LABEL_B = 1
    LABEL_I = 2

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[torch.device] = None,
        max_length: int = 512,
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.model.to(self.device)
        self.model.eval()

    def detect_and_sanitize(
        self,
        text: str,
        redaction_marker: str = "[REDACTED_INJECTION_PAYLOAD]",
    ) -> SanitizationResult:
        """
        Extracts injection spans from input text and provides sanitized/redacted outputs.
        """
        if not text or not text.strip():
            return SanitizationResult(
                original_text=text,
                sanitized_text=text,
                has_injection=False,
                spans=[],
                redacted_text=text,
            )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # (seq_len, 3)
            probs = F.softmax(logits, dim=-1)

        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        # Extract contiguous spans where prediction is B or I
        spans: List[InjectionSpan] = []
        current_span_start: Optional[int] = None
        current_span_end: Optional[int] = None
        current_span_probs: List[float] = []

        for idx, (pred, (tok_start, tok_end)) in enumerate(zip(preds, offset_mapping)):
            if tok_start == 0 and tok_end == 0:
                continue  # Special token

            attack_prob = (probs[idx, self.LABEL_B] + probs[idx, self.LABEL_I]).item()

            if pred in (self.LABEL_B, self.LABEL_I) and attack_prob >= self.confidence_threshold:
                if current_span_start is None:
                    current_span_start = tok_start
                    current_span_end = tok_end
                    current_span_probs = [attack_prob]
                else:
                    current_span_end = max(current_span_end, tok_end)
                    current_span_probs.append(attack_prob)
            else:
                if current_span_start is not None and current_span_end is not None:
                    mean_conf = sum(current_span_probs) / max(len(current_span_probs), 1)
                    span_text = text[current_span_start:current_span_end]
                    spans.append(
                        InjectionSpan(
                            start_char=current_span_start,
                            end_char=current_span_end,
                            text=span_text,
                            confidence=mean_conf,
                        )
                    )
                    current_span_start = None
                    current_span_end = None
                    current_span_probs = []

        # Flush trailing span
        if current_span_start is not None and current_span_end is not None:
            mean_conf = sum(current_span_probs) / max(len(current_span_probs), 1)
            span_text = text[current_span_start:current_span_end]
            spans.append(
                InjectionSpan(
                    start_char=current_span_start,
                    end_char=current_span_end,
                    text=span_text,
                    confidence=mean_conf,
                )
            )

        # Build sanitized and redacted strings by slicing
        sanitized_chunks = []
        redacted_chunks = []
        last_idx = 0

        for span in spans:
            sanitized_chunks.append(text[last_idx:span.start_char])
            redacted_chunks.append(text[last_idx:span.start_char])
            redacted_chunks.append(redaction_marker)
            last_idx = span.end_char

        sanitized_chunks.append(text[last_idx:])
        redacted_chunks.append(text[last_idx:])

        sanitized_text = "".join(sanitized_chunks).strip()
        redacted_text = "".join(redacted_chunks).strip()

        return SanitizationResult(
            original_text=text,
            sanitized_text=sanitized_text,
            has_injection=len(spans) > 0,
            spans=spans,
            redacted_text=redacted_text,
        )
