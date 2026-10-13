"""
v2/demo/server.py

FastAPI interactive backend for Adaptive-IPI v2:
- Performs Token-Level Span Extraction & Sanitization using ModernBERT Hardened model.
- Provides side-by-side comparative analytics vs. v1 binary classification.
- Serves the interactive web UI.
"""

import os
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.inference.sanitizer import IPISanitizer

app = FastAPI(title="Adaptive-IPI v2: Span Extraction & Sanitizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Model paths
V2_MODEL_PATH = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_enterprise" / "best"
if not V2_MODEL_PATH.exists():
    V2_MODEL_PATH = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_adaptive" / "best"
if not V2_MODEL_PATH.exists():
    V2_MODEL_PATH = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_hardened" / "best"
if not V2_MODEL_PATH.exists():
    V2_MODEL_PATH = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_robust" / "best"

V1_MODEL_PATH = _PROJECT_ROOT / "demo" / "models" / "adaptive_ipi_model"

v2_sanitizer: Optional[IPISanitizer] = None
v1_tokenizer = None
v1_model = None


@app.on_event("startup")
def load_models():
    global v2_sanitizer, v1_tokenizer, v1_model
    print(f"Loading v2 model from {V2_MODEL_PATH} onto {DEVICE}...")
    v2_tokenizer = AutoTokenizer.from_pretrained(V2_MODEL_PATH)
    v2_model = AutoModelForTokenClassification.from_pretrained(V2_MODEL_PATH)
    v2_sanitizer = IPISanitizer(v2_model, v2_tokenizer, device=DEVICE)
    print("✅ v2 Span Sanitizer loaded successfully.")

    if V1_MODEL_PATH.exists():
        print(f"Loading v1 baseline model from {V1_MODEL_PATH} onto {DEVICE}...")
        try:
            v1_tokenizer = AutoTokenizer.from_pretrained(V1_MODEL_PATH)
            v1_model = AutoModelForSequenceClassification.from_pretrained(V1_MODEL_PATH)
            v1_model.to(DEVICE)
            v1_model.eval()
            print("✅ v1 Baseline Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Could not load v1 baseline: {e}")


class SanitizeRequest(BaseModel):
    text: str
    threshold: float = 0.5


class SpanDTO(BaseModel):
    start_char: int
    end_char: int
    text: str
    confidence: float


class SanitizeResponse(BaseModel):
    original_text: str
    sanitized_text: str
    redacted_text: str
    has_injection: bool
    spans: List[SpanDTO]
    latency_ms: float
    retention_rate: float
    # Comparison metrics
    v1_prediction: str
    v1_confidence: float
    v1_action: str
    v2_action: str


@app.post("/api/sanitize", response_model=SanitizeResponse)
async def sanitize_endpoint(req: SanitizeRequest):
    if not v2_sanitizer:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    t0 = time.perf_counter()
    v2_result = v2_sanitizer.detect_and_sanitize(req.text)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate data retention rate
    orig_len = len(req.text.strip())
    san_len = len(v2_result.sanitized_text.strip())
    retention_rate = (san_len / max(orig_len, 1)) * 100.0 if v2_result.has_injection else 100.0

    # Run v1 model if available for side-by-side comparison
    v1_label = "SAFE"
    v1_conf = 0.0
    v1_action = "PASSED_TO_LLM"

    if v1_model and v1_tokenizer:
        with torch.no_grad():
            inputs = v1_tokenizer(
                req.text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            logits = v1_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
            attack_p = probs[1].item()
            benign_p = probs[0].item()

            if attack_p >= 0.5:
                v1_label = "ATTACK_DETECTED"
                v1_conf = attack_p
                v1_action = "DISCARD_ENTIRE_EMAIL (100% Data Loss)"
            else:
                v1_label = "SAFE"
                v1_conf = benign_p
                v1_action = "PASSED_TO_LLM"
    else:
        # Fallback simulation if v1 weights not loaded
        if v2_result.has_injection:
            v1_label = "ATTACK_DETECTED"
            v1_conf = 0.95
            v1_action = "DISCARD_ENTIRE_EMAIL (100% Data Loss)"

    v2_action = (
        f"SURGICAL_EXCISION ({retention_rate:.1f}% Legitimate Data Preserved)"
        if v2_result.has_injection
        else "PASSED_UNMODIFIED (100% Context Preserved)"
    )

    return SanitizeResponse(
        original_text=req.text,
        sanitized_text=v2_result.sanitized_text,
        redacted_text=v2_result.redacted_text,
        has_injection=v2_result.has_injection,
        spans=[
            SpanDTO(
                start_char=s.start_char,
                end_char=s.end_char,
                text=s.text,
                confidence=s.confidence,
            )
            for s in v2_result.spans
        ],
        latency_ms=latency_ms,
        retention_rate=retention_rate,
        v1_prediction=v1_label,
        v1_confidence=v1_conf,
        v1_action=v1_action,
        v2_action=v2_action,
    )


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "v2_model": str(V2_MODEL_PATH.name),
        "device": str(DEVICE),
        "v1_loaded": v1_model is not None,
    }


# Mount static web assets
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


def start_server(port: int = 8080):
    import uvicorn
    uvicorn.run("v2.demo.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    start_server()
