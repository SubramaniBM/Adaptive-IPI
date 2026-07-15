import os
import time
import re
from contextlib import asynccontextmanager
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "../models/adaptive_ipi_model")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

# ---------------------------------------------------------
# Explainability Rules
# ---------------------------------------------------------
EXPLAINABILITY_RULES = [
    (re.compile(r"ignore previous instructions", re.IGNORECASE), "Possible instruction override detected."),
    (re.compile(r"system prompt", re.IGNORECASE), "Possible prompt extraction attempt."),
    (re.compile(r"reveal secrets", re.IGNORECASE), "Sensitive information request detected."),
    (re.compile(r"you are now", re.IGNORECASE), "Possible persona adoption / jailbreak attempt."),
    (re.compile(r"disregard", re.IGNORECASE), "Possible instruction override detected."),
]

# ---------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model exactly once at startup."""
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down model...")

# ---------------------------------------------------------
# API Setup
# ---------------------------------------------------------
app = FastAPI(title="Adaptive-IPI Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    explain: bool = False

class AttributionToken(BaseModel):
    token: str
    score: float

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    attack_probability: float
    benign_probability: float
    inference_time_ms: float
    reason: str
    attributions: Optional[List[AttributionToken]] = None

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    start_time = time.perf_counter()
    
    # 1. Rule-based Explainability
    reasons = []
    for pattern, reason in EXPLAINABILITY_RULES:
        if pattern.search(req.text):
            reasons.append(reason)
            
    # 2. Tokenization
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
    # Assuming class 0 = Benign, class 1 = Attack
    benign_prob = probs[0, 0].item()
    attack_prob = probs[0, 1].item()
    
    predicted_class = 1 if attack_prob >= 0.5 else 0
    prediction_label = "Attack" if predicted_class == 1 else "SAFE"
    confidence = attack_prob if predicted_class == 1 else benign_prob
    
    # 4. Integrated Gradients Explainability (Optional)
    attributions_list = None
    if req.explain:
        # Construct baseline (zero tokens except CLS/SEP)
        input_ids = inputs["input_ids"]
        ref_input_ids = torch.zeros_like(input_ids).to(DEVICE)
        ref_input_ids[0, 0] = tokenizer.cls_token_id
        ref_input_ids[0, -1] = tokenizer.sep_token_id
        
        def forward_func(inputs, attention_mask=None):
            return model(inputs, attention_mask=attention_mask).logits
            
        # Use tok_embeddings for ModernBERT
        lig = LayerIntegratedGradients(forward_func, model.model.embeddings.tok_embeddings)
        
        attrs, _ = lig.attribute(inputs=input_ids,
                                baselines=ref_input_ids,
                                additional_forward_args=(inputs["attention_mask"],),
                                target=1, # Calculate attribution pushing towards Attack
                                return_convergence_delta=True)
                                
        attrs = attrs.sum(dim=-1).squeeze(0)
        norm_factor = torch.norm(attrs)
        if norm_factor > 0:
            attrs = attrs / norm_factor
            
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        attributions_list = []
        for token, attr in zip(tokens, attrs.cpu().detach().numpy()):
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                clean_token = token.replace('Ġ', ' ') if token.startswith('Ġ') else token
                attributions_list.append(AttributionToken(token=clean_token, score=float(attr)))
    
    end_time = time.perf_counter()
    inference_time_ms = round((end_time - start_time) * 1000, 2)
    
    # Combine reasons or provide default
    if predicted_class == 1:
        if not reasons:
            reasons.append("Detected suspicious injection patterns (Neural Net activation).")
        final_reason = " ".join(reasons)
    else:
        final_reason = "No malicious patterns detected."

    return PredictResponse(
        prediction=prediction_label,
        confidence=confidence,
        attack_probability=attack_prob,
        benign_probability=benign_prob,
        inference_time_ms=inference_time_ms,
        reason=final_reason,
        attributions=attributions_list
    )
