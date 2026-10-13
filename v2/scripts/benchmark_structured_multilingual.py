"""
v2/scripts/benchmark_structured_multilingual.py

Benchmarks the current Adaptive-IPI span detector on:
1. Structured RAG Formats (CSV tables, JSON blocks, Code docstrings)
2. Multilingual & Polyglot Evasion Vectors (Spanish, French, German, Chinese, Russian)
Measures Detection Rate, Span IoU, and Benign False Alarm Rate per subcategory.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForTokenClassification
from v2.src.adversarial.structured_multilingual_vectors import StructuredMultilingualGenerator
from v2.src.adversarial.hard_negative_miner import HardNegativeMiner


def main():
    print("=" * 75)
    print("  ADAPTIVE-IPI: STRUCTURED RAG & MULTILINGUAL BENCHMARK EVALUATOR  ")
    print("=" * 75)

    ckpt_path = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_enterprise" / "best"
    if not ckpt_path.exists():
        ckpt_path = _PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_adaptive" / "best"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Model Checkpoint: {ckpt_path.relative_to(_PROJECT_ROOT)}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
    model = AutoModelForTokenClassification.from_pretrained(str(ckpt_path))
    model.to(device)

    miner = HardNegativeMiner(model, tokenizer, device, threshold=0.5)
    generator = StructuredMultilingualGenerator()

    # Define test categories
    categories = [
        ("Structured CSV Ingestion", [generator.generate_structured_injection("csv") for _ in range(30)], [generator.generate_structured_benign("csv") for _ in range(30)]),
        ("Structured JSON Payloads", [generator.generate_structured_injection("json") for _ in range(30)], [generator.generate_structured_benign("json") for _ in range(30)]),
        ("Structured Code / Docstrings", [generator.generate_structured_injection("code") for _ in range(30)], [generator.generate_structured_benign("code") for _ in range(30)]),
        ("Multilingual: Spanish (ES)", [generator.generate_multilingual_injection("es") for _ in range(30)], [generator.generate_multilingual_benign("es") for _ in range(30)]),
        ("Multilingual: French (FR)", [generator.generate_multilingual_injection("fr") for _ in range(30)], [generator.generate_multilingual_benign("fr") for _ in range(30)]),
        ("Multilingual: German (DE)", [generator.generate_multilingual_injection("de") for _ in range(30)], [generator.generate_multilingual_benign("de") for _ in range(30)]),
        ("Multilingual: Chinese (ZH)", [generator.generate_multilingual_injection("zh") for _ in range(30)], [generator.generate_multilingual_benign("zh") for _ in range(30)]),
        ("Multilingual: Russian (RU)", [generator.generate_multilingual_injection("ru") for _ in range(30)], [generator.generate_multilingual_benign("ru") for _ in range(30)]),
    ]

    print("\nRunning diagnostic evaluation across 8 enterprise categories (480 total samples)...\n")
    print(f"{'Threat Category':<30} | {'Det. Rate (%)':<14} | {'Mean IoU (%)':<13} | {'Benign FPR (%)':<14}")
    print("-" * 78)

    results = []
    for cat_name, attacks, benigns in categories:
        combined = attacks + benigns
        _, metrics = miner.evaluate_candidates(combined, batch_size=16)

        det_rate = 100.0 - metrics["attack_success_rate"]
        iou = metrics["mean_iou"]
        fpr = metrics["false_positive_rate"]

        results.append({
            "category": cat_name,
            "det_rate": det_rate,
            "mean_iou": iou,
            "fpr": fpr,
        })

        print(f"{cat_name:<30} | {det_rate:>13.1f}% | {iou:>12.1f}% | {fpr:>13.1f}%")

    print("=" * 78)

    overall_det = np.mean([r["det_rate"] for r in results])
    overall_iou = np.mean([r["mean_iou"] for r in results])
    overall_fpr = np.mean([r["fpr"] for r in results])
    print(f"{'OVERALL AVERAGE':<30} | {overall_det:>13.1f}% | {overall_iou:>12.1f}% | {overall_fpr:>13.1f}%\n")


if __name__ == "__main__":
    main()
