"""
v2/scripts/export_onnx.py

Exports the trained ModernBERT Span Extraction model to ONNX
and applies INT8 Dynamic Quantization for sub-5ms CPU execution.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def export_model():
    model_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_enterprise" / "best"
    if not model_dir.exists():
        model_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_adaptive" / "best"
    if not model_dir.exists():
        model_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_hardened" / "best"
    if not model_dir.exists():
        model_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_robust" / "best"

    output_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = output_dir / "model.onnx"
    int8_onnx_path = output_dir / "model_int8.onnx"

    print(f"Loading PyTorch model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    model.to("cpu")

    # Save tokenizer to onnx directory for standalone distribution
    tokenizer.save_pretrained(output_dir)

    print("Creating dummy input for tracing...")
    dummy_text = "SUBJECT: Account Alert. Please verify the billing transaction reference."
    dummy_inputs = tokenizer(
        dummy_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    print(f"Exporting to ONNX FP32: {fp32_onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        f=str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    print(f"✅ ONNX FP32 model created: {fp32_size_mb:.1f} MB")

    print(f"\nApplying INT8 Dynamic Quantization: {int8_onnx_path}...")
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    compression_ratio = fp32_size_mb / max(int8_size_mb, 1)

    print("=" * 70)
    print("🎉 Model Export & Quantization Complete!")
    print(f"  FP32 ONNX Size:        {fp32_size_mb:.1f} MB")
    print(f"  INT8 Quantized Size:   {int8_size_mb:.1f} MB")
    print(f"  Memory Footprint Reduction: {compression_ratio:.1f}x smaller")
    print(f"  Artifact location:     {int8_onnx_path}")
    print("=" * 70)


if __name__ == "__main__":
    export_model()
