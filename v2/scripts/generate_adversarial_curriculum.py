"""
v2/scripts/generate_adversarial_curriculum.py

Synthesizes a modern adversarial curriculum covering 4 real-world attack vectors:
1. Markdown Exfiltration
2. Delimiter & HTML Smuggling
3. Context-Blended Social Engineering
4. Syntactic Obfuscation (Base64, Zero-width, Leetspeak)

Merges these into the robust training data to produce:
v2/data/processed/train_hardened.csv
"""

from pathlib import Path
import random
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys_path = str(_PROJECT_ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from v2.src.adversarial.threat_vectors import AdversarialVectorGenerator


def main():
    generator = AdversarialVectorGenerator()
    robust_train_path = _PROJECT_ROOT / "v2" / "data" / "processed" / "train_robust.csv"
    output_path = _PROJECT_ROOT / "v2" / "data" / "processed" / "train_hardened.csv"

    print("Loading base training data...")
    robust_df = pd.read_csv(robust_train_path)

    # Use clean benign contexts from the training set
    benign_pool = robust_df[robust_df["label"] == 0]["context"].dropna().tolist()
    print(f"Benign context pool size: {len(benign_pool):,}")

    categories = ["markdown_exfil", "delimiter_smuggling", "context_blended", "obfuscated"]
    samples_per_cat = 150  # 600 total modern attack vectors

    adversarial_rows = []
    print(f"Generating {len(categories) * samples_per_cat} modern adversarial injection samples...")

    for cat in categories:
        for i in range(samples_per_cat):
            ctx = random.choice(benign_pool)
            sample = generator.inject_adversarial_payload(ctx, category=cat)

            adversarial_rows.append({
                "id": f"adv_{cat}_{i:04d}",
                "context": sample["text"],
                "attack_instruction": sample["payload"],
                "label": 1,
                "attack_family": cat,
                "attack_position": sample["position"],
                "split": "train",
                "task": "adversarial_hardened",
                "source": f"adversarial_{cat}",
                "span_start": sample["span_start"],
                "span_end": sample["span_end"],
            })

    adv_df = pd.DataFrame(adversarial_rows)
    hardened_df = pd.concat([robust_df, adv_df], ignore_index=True)

    hardened_df.to_csv(output_path, index=False)
    print("=" * 70)
    print("✅ Adversarial Curriculum Generation Complete!")
    print(f"  Base robust train size:      {len(robust_df):,}")
    print(f"  Added modern evasion vectors: {len(adv_df):,} (150 per category)")
    print(f"  Final hardened train size:    {len(hardened_df):,}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
