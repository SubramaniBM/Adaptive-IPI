"""
v2/scripts/prepare_span_data.py

Precomputes character-level span boundaries for training and evaluation.
"""

from pathlib import Path
import json
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def process_split(input_csv: Path, output_csv: Path) -> dict:
    df = pd.read_csv(input_csv)
    
    span_starts = []
    span_ends = []
    matched_count = 0
    attack_count = 0

    for _, row in df.iterrows():
        is_attack = int(row["label"]) == 1
        text = str(row["context"])
        instr = str(row.get("attack_instruction", ""))

        if is_attack:
            attack_count += 1
            if instr and instr in text:
                s = text.find(instr)
                e = s + len(instr)
                span_starts.append(s)
                span_ends.append(e)
                matched_count += 1
            else:
                span_starts.append(-1)
                span_ends.append(-1)
        else:
            span_starts.append(-1)
            span_ends.append(-1)

    df["span_start"] = span_starts
    df["span_end"] = span_ends

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    stats = {
        "total_samples": len(df),
        "attack_samples": attack_count,
        "benign_samples": len(df) - attack_count,
        "matched_spans": matched_count,
        "match_percentage": (matched_count / max(attack_count, 1)) * 100.0,
    }
    return stats


def main():
    raw_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    out_dir = _PROJECT_ROOT / "v2" / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ["train", "validation", "test"]:
        in_path = raw_dir / f"{split}.csv"
        out_path = out_dir / f"{split}.csv"
        print(f"Processing {split} split from {in_path}...")
        stats = process_split(in_path, out_path)
        summary[split] = stats
        print(f"  {split}: {stats['matched_spans']}/{stats['attack_samples']} attack spans localized ({stats['match_percentage']:.1f}%)")

    with open(out_dir / "span_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSpan dataset preparation complete! Files written to {out_dir}")


if __name__ == "__main__":
    main()
