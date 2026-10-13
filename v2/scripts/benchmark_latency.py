"""
v2/scripts/benchmark_latency.py

Benchmarks CPU inference latency across:
1. PyTorch CPU (FP32)
2. ONNX Runtime CPU (FP32)
3. ONNX Runtime CPU (INT8 Quantized)

Measures P50, P95, P99 latency and document throughput per second.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.inference.onnx_sanitizer import ONNXIPISanitizer


def benchmark_engine(name: str, fn, texts: list, warmup: int = 10):
    # Warmup
    for text in texts[:warmup]:
        fn(text)

    latencies = []
    for text in texts:
        t0 = time.perf_counter()
        fn(text)
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

    lats = np.array(latencies)
    mean_lat = np.mean(lats)
    p50 = np.percentile(lats, 50)
    p95 = np.percentile(lats, 95)
    p99 = np.percentile(lats, 99)
    throughput = 1000.0 / mean_lat

    return {
        "engine": name,
        "mean_ms": mean_lat,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "throughput_docs_per_sec": throughput,
    }


def main():
    onnx_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "onnx"
    pytorch_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_hardened" / "best"
    if not pytorch_dir.exists():
        pytorch_dir = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_robust" / "best"

    # Load 100 test documents
    test_csv = _PROJECT_ROOT / "v2" / "data" / "processed" / "test.csv"
    df = pd.read_csv(test_csv)
    texts = df["context"].dropna().astype(str).tolist()[:100]

    print(f"Benchmarking CPU Latency across {len(texts)} real email documents...\n")

    # 1. PyTorch CPU
    print("Loading PyTorch CPU model...")
    pt_tokenizer = AutoTokenizer.from_pretrained(pytorch_dir)
    pt_model = AutoModelForTokenClassification.from_pretrained(pytorch_dir)
    pt_model.eval()
    pt_model.to("cpu")

    def run_pytorch(text):
        inputs = pt_tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        with torch.no_grad():
            pt_model(**inputs)

    res_pt = benchmark_engine("PyTorch CPU (FP32)", run_pytorch, texts)

    # 2. ONNX FP32
    print("Loading ONNX FP32 engine...")
    onnx_fp32 = ONNXIPISanitizer.load(onnx_dir, use_int8=False)
    res_fp32 = benchmark_engine("ONNX Runtime (FP32)", onnx_fp32.detect_and_sanitize, texts)

    # 3. ONNX INT8 Quantized
    print("Loading ONNX INT8 Quantized engine...")
    onnx_int8 = ONNXIPISanitizer.load(onnx_dir, use_int8=True)
    res_int8 = benchmark_engine("ONNX Runtime (INT8)", onnx_int8.detect_and_sanitize, texts)

    # Print Table
    results = [res_pt, res_fp32, res_int8]
    print("\n" + "=" * 90)
    print(f"{'Engine':<25} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10} | {'Throughput':<15}")
    print("-" * 90)
    for r in results:
        print(f"{r['engine']:<25} | {r['mean_ms']:<10.2f} | {r['p50_ms']:<10.2f} | {r['p95_ms']:<10.2f} | {r['p99_ms']:<10.2f} | {r['throughput_docs_per_sec']:<8.1f} docs/s")
    print("=" * 90)


if __name__ == "__main__":
    main()
