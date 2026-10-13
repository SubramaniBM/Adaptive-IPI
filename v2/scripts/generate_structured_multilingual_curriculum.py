"""
v2/scripts/generate_structured_multilingual_curriculum.py

Synthesizes a balanced Enterprise RAG & Multilingual Curriculum dataset:
- Injected CSV tables, JSON objects, Code docstrings
- Clean benign CSV tables, JSON configs, and Code snippets
- Multilingual polyglot injections (ES, FR, DE, ZH, RU)
- Clean benign multilingual business communications (ES, FR, DE, ZH, RU)
Merges with replay buffer to create `v2/data/processed/train_enterprise.csv`.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from v2.src.adversarial.structured_multilingual_vectors import StructuredMultilingualGenerator


def main():
    print("Synthesizing Enterprise Structured & Multilingual Curriculum...")
    generator = StructuredMultilingualGenerator()

    # Generate 50 samples per subcategory
    suite = generator.generate_comprehensive_suite(samples_per_subcat=50)
    print(f"Generated {len(suite)} structured and multilingual samples.")

    rows = []
    for s in suite:
        rows.append({
            "context": s["text"],
            "label": s["label"],
            "attack_instruction": s["payload"] if s["label"] == 1 else "",
            "category": s["category"],
        })

    new_df = pd.DataFrame(rows)

    # Load existing hardened dataset to maintain baseline abilities (replay buffer)
    data_dir = _PROJECT_ROOT / "v2" / "data" / "processed"
    replay_file = data_dir / "curriculum_round_2.csv"
    if not replay_file.exists():
        replay_file = data_dir / "train_hardened.csv"

    replay_df = pd.read_csv(replay_file)
    if "category" not in replay_df.columns:
        replay_df["category"] = "replay_buffer"

    # Combine new enterprise suite with replay buffer
    combined_df = pd.concat([new_df, replay_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_file = data_dir / "train_enterprise.csv"
    combined_df.to_csv(out_file, index=False)

    print(f"Successfully compiled enterprise training dataset: {len(combined_df)} samples at {out_file.relative_to(_PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
