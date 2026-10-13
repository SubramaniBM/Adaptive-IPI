"""
v2/scripts/mine_benign_imperatives.py

Mines hard-negative benign enterprise communications containing imperative language
(e.g., "please", "review", "forward", "confirm", "approve") to teach the token classifier
the boundary between legitimate business requests and malicious system overrides.
"""

from pathlib import Path
import re
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    raw_path = _PROJECT_ROOT / "data" / "raw" / "enron_spam.csv"
    heldout_path = _PROJECT_ROOT / "data" / "processed" / "enron_heldout_test.csv"
    v2_train_path = _PROJECT_ROOT / "v2" / "data" / "processed" / "train.csv"
    output_path = _PROJECT_ROOT / "v2" / "data" / "processed" / "train_robust.csv"

    print("Loading datasets...")
    raw = pd.read_csv(raw_path)
    ham = raw[raw["label_text"] == "ham"].dropna(subset=["text"])

    heldout = pd.read_csv(heldout_path)
    heldout_texts = set(heldout["context"].astype(str))

    # Strict zero-leakage check: drop any text present in held-out test set
    candidates = ham[~ham["text"].isin(heldout_texts)].copy()
    print(f"Candidates after zero-leakage test filtering: {len(candidates):,}")

    # Regex for common imperative business verbs
    imperative_pattern = re.compile(
        r"\b(please|review|forward|approve|confirm|attached|schedule|verify|submit|notify|ensure|update|action required|draft)\b",
        re.IGNORECASE,
    )

    valid = candidates[
        candidates["text"].apply(
            lambda t: bool(imperative_pattern.search(t)) and 60 < len(t.strip()) < 1200
        )
    ].copy()

    # Sample 500 clean, diverse samples
    mined = valid.sample(n=500, random_state=42).reset_index(drop=True)

    # Format according to v2 dataset schema
    mined_df = pd.DataFrame({
        "id": [f"mined_imperative_{i:04d}" for i in range(len(mined))],
        "context": mined["text"].values,
        "attack_instruction": [""] * len(mined),
        "label": [0] * len(mined),
        "attack_family": ["None"] * len(mined),
        "attack_position": ["None"] * len(mined),
        "split": ["train"] * len(mined),
        "task": ["enterprise_email"] * len(mined),
        "source": ["enron_mined_imperative"] * len(mined),
        "span_start": [-1] * len(mined),
        "span_end": [-1] * len(mined),
    })

    # Load existing train set and merge
    train_orig = pd.read_csv(v2_train_path)
    robust_train = pd.concat([train_orig, mined_df], ignore_index=True)

    robust_train.to_csv(output_path, index=False)
    print(f"✅ Mined 500 benign imperative hard-negatives.")
    print(f"Original train size: {len(train_orig):,} (attacks: {sum(train_orig['label']==1)}, benign: {sum(train_orig['label']==0)})")
    print(f"Robust train size:   {len(robust_train):,} (attacks: {sum(robust_train['label']==1)}, benign: {sum(robust_train['label']==0)})")
    print(f"Saved robust training set to: {output_path}")


if __name__ == "__main__":
    main()
