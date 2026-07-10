import pandas as pd
import json
import random
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.utils.reproducibility import set_seed

def main():
    set_seed(42)
    dataset_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    
    train_df = pd.read_csv(dataset_dir / "train.csv")
    val_df = pd.read_csv(dataset_dir / "validation.csv")
    test_df = pd.read_csv(dataset_dir / "test.csv")
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Task 1: Dataset Statistics
    stats = {}
    for split_name, df in zip(["TRAIN", "VALIDATION", "TEST"], [train_df, val_df, test_df]):
        stats[split_name] = {
            "total": len(df),
            "benign": len(df[df["label"] == 0]),
            "attack": len(df[df["label"] == 1])
        }
    stats["TOTAL"] = {
        "total": len(all_df),
        "benign": len(all_df[all_df["label"] == 0]),
        "attack": len(all_df[all_df["label"] == 1])
    }
    
    # Task 2: Split Integrity
    train_contexts = set(train_df["context"].unique())
    val_contexts = set(val_df["context"].unique())
    test_contexts = set(test_df["context"].unique())
    
    integrity = {
        "Train ∩ Validation": len(train_contexts & val_contexts),
        "Train ∩ Test": len(train_contexts & test_contexts),
        "Validation ∩ Test": len(val_contexts & test_contexts)
    }
    
    # Task 3: Duplicate Audit
    duplicates = {
        "duplicate_ids": int(all_df["id"].duplicated().sum()),
        "duplicate_context_intent_pairs": int(all_df.duplicated(subset=["context", "attack_instruction"]).sum()),
        "duplicate_attack_samples": int(all_df[all_df["label"] == 1].duplicated(subset=["context", "attack_instruction"]).sum()),
        "duplicate_benign_samples": int(all_df[all_df["label"] == 0].duplicated(subset=["context", "attack_instruction"]).sum()),
    }
    
    # Task 6: Attack Coverage
    attacks = all_df[all_df["label"] == 1]
    attack_family_counts = attacks["attack_family"].value_counts().to_dict()
    attack_position_counts = attacks["attack_position"].value_counts().to_dict()
    coverage = {
        "families": attack_family_counts,
        "positions": attack_position_counts
    }
    
    # Task 4: Random Manual Inspection
    benign_samples = all_df[all_df["label"] == 0].sample(20, random_state=42)
    attack_samples = all_df[all_df["label"] == 1].sample(20, random_state=42)
    
    def format_sample(row):
        return {
            "id": row["id"],
            "label": row["label"],
            "context": str(row["context"])[:100] + "...",
            "user_intent": row["attack_instruction"],
            "attack_family": row.get("attack_family", "N/A"),
            "attack_position": row.get("attack_position", "N/A")
        }
    
    inspection = {
        "benign": [format_sample(row) for _, row in benign_samples.iterrows()],
        "attack": [format_sample(row) for _, row in attack_samples.iterrows()]
    }
    
    # Task 5: Generated Benign Quality
    # 100 randomly selected benign intents
    benign_intents = all_df[all_df["label"] == 0]["attack_instruction"].sample(100, random_state=42).tolist()
    
    output = {
        "stats": stats,
        "integrity": integrity,
        "duplicates": duplicates,
        "coverage": coverage,
        "inspection": inspection,
        "benign_intents": benign_intents
    }
    
    with open(str(_PROJECT_ROOT / "reports" / "audit_data.json"), "w") as f:
        json.dump(output, f, indent=2)
    print("Done")

if __name__ == "__main__":
    main()
