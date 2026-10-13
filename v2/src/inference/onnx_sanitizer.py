"""
v2/src/inference/onnx_sanitizer.py

Ultra-fast, production-grade CPU inference engine using ONNX Runtime (INT8).
Enables sub-5ms token classification, span extraction, and document sanitization.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from v2.src.core.types import InjectionSpan, SanitizationResult


class ONNXIPISanitizer:
    """
    High-throughput CPU guardrail for real-time RAG ingestion pipelines.
    Runs INT8 quantized ONNX session with sub-5ms latency.
    """

    LABEL_O = 0
    LABEL_B = 1
    LABEL_I = 2

    def __init__(
        self,
        onnx_model_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 256,
        confidence_threshold: float = 0.5,
        num_threads: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU Execution Provider
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    @classmethod
    def load(
        cls,
        model_dir: Path,
        use_int8: bool = True,
        confidence_threshold: float = 0.5,
    ) -> "ONNXIPISanitizer":
        model_filename = "model_int8.onnx" if use_int8 else "model.onnx"
        model_path = model_dir / model_filename
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return cls(
            onnx_model_path=model_path,
            tokenizer=tokenizer,
            confidence_threshold=confidence_threshold,
        )

    def detect_and_sanitize(
        self,
        text: str,
        redaction_marker: str = "[REDACTED_INJECTION_PAYLOAD]",
    ) -> SanitizationResult:
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
        )

        input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
        attention_mask = np.array([encoding["attention_mask"]], dtype=np.int64)
        offset_mapping = encoding["offset_mapping"]

        # ONNX inference
        outputs = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        logits = outputs[0][0]  # (seq_len, 3)

        # Softmax on CPU
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        preds = np.argmax(logits, axis=-1)

        spans: List[InjectionSpan] = []
        current_span_start: Optional[int] = None
        current_span_end: Optional[int] = None
        current_span_probs: List[float] = []

        for idx, (pred, (tok_start, tok_end)) in enumerate(zip(preds, offset_mapping)):
            if tok_start == 0 and tok_end == 0:
                continue

            attack_prob = float(probs[idx, self.LABEL_B] + probs[idx, self.LABEL_I])

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

        # Surgical excision
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
