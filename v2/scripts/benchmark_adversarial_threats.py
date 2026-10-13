"""
v2/scripts/benchmark_adversarial_threats.py

Benchmarks the trained Span Detector against modern 2025-2026 evasion vectors:
1. Markdown Exfiltration
2. Delimiter & HTML Smuggling
3. Context-Blended Social Engineering
4. Syntactic Obfuscation (Base64, Leetspeak, Zero-width)
"""

import sys
from pathlib import Path
import random
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.adversarial.threat_vectors import AdversarialVectorGenerator
from v2.src.inference.sanitizer import IPISanitizer
from v2.src.evaluation.span_metrics import compute_span_character_iou


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_PROJECT_ROOT / "v2" / "checkpoints" / "span_modernbert_robust" / "best",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    model_path = args.model_path

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model from {model_path} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    sanitizer = IPISanitizer(model, tokenizer, device=device)

    # Load authentic benign contexts from Enron held-out test to inject into
    heldout = pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "enron_heldout_test.csv")
    contexts = [str(t) for t in heldout["context"].dropna().tolist() if 100 < len(str(t)) < 600][:100]

    generator = AdversarialVectorGenerator()
    categories = ["markdown_exfil", "delimiter_smuggling", "context_blended", "obfuscated"]
    samples_per_cat = 25

    results = {}
    print("\n" + "=" * 80)
    print("BENCHMARKING MODEL ON MODERN ADVERSARIAL THREAT VECTORS")
    print("=" * 80)

    for cat in categories:
        detected = 0
        ious = []

        for _ in range(samples_per_cat):
            ctx = random.choice(contexts)
            item = generator.inject_adversarial_payload(ctx, category=cat)

            res = sanitizer.detect_and_sanitize(item["text"])

            if res.has_injection and len(res.spans) > 0:
                detected += 1
                best_iou = max(
                    compute_span_character_iou(
                        (s.start_char, s.end_char),
                        (item["span_start"], item["span_end"]),
                    )
                    for s in res.spans
                )
                ious.append(best_iou)
            else:
                ious.append(0.0)

        det_rate = (detected / samples_per_cat) * 100.0
        mean_iou = (sum(ious) / len(ious)) * 100.0
        results[cat] = {
            "detection_rate": det_rate,
            "mean_iou": mean_iou,
            "detected": detected,
            "total": samples_per_cat,
        }

        print(f"Category: {cat:25s} | Detection Rate: {det_rate:5.1f}% | Mean Span IoU: {mean_iou:5.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
