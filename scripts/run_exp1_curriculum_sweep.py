"""
scripts/run_exp1_curriculum_sweep.py

Experiment 1: Curriculum Size Sweep (Ablation)
Generates partial curriculums (N=20, 40, 60) from the 91 adversarial samples
and retrains the student model sequentially to observe the degradation gradient.
"""

import sys
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def create_partial_curriculum(n_samples: int, version: int):
    print(f"\n--- Creating dataset_v{version} with {n_samples} curriculum samples ---")
    source_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    target_dir = _PROJECT_ROOT / "data" / "processed" / f"dataset_v{version}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy validation and test
    shutil.copy(source_dir / "validation.csv", target_dir / "validation.csv")
    if (source_dir / "test.csv").exists():
        shutil.copy(source_dir / "test.csv", target_dir / "test.csv")
        
    # Read V0 train
    source_train = pd.read_csv(source_dir / "train.csv")
    
    # Read full curriculum (91 samples) from Phase 5 (exp027)
    curr_path = _PROJECT_ROOT / "experiments" / "exp027" / "generated_curriculum.jsonl"
    curr_data = []
    with open(curr_path, "r", encoding="utf-8") as f:
        for line in f:
            curr_data.append(json.loads(line))
            
    # Take first N samples
    curr_data = curr_data[:n_samples]
    
    curr_df = pd.DataFrame(curr_data)
    curr_df["id"] = [f"curr_{i}" for i in range(len(curr_df))]
    
    target_train = pd.concat([source_train, curr_df], ignore_index=True)
    target_train.to_csv(target_dir / "train.csv", index=False)
    print(f"Saved {len(target_train)} total samples to {target_dir / 'train.csv'}")


def run_pipeline(version: int):
    print(f"\n--- Running Training for dataset_v{version} ---")
    
    # Train
    train_cmd = [
        "python", "scripts/run_phase3_distill.py",
        "--student-config", "configs/mac_student.yaml",
        "--data-config", "configs/mac_data.yaml",
        "--dataset-version", str(version)
    ]
    subprocess.run(train_cmd, check=True)
    
    print(f"\n--- Running Analysis for dataset_v{version} ---")
    # We must find the latest experiment directory created by training
    import glob
    import os
    exp_dirs = sorted(glob.glob(str(_PROJECT_ROOT / "outputs/mac_experiments/exp*")), key=os.path.getmtime)
    latest_exp = exp_dirs[-1]
    checkpoint_path = f"{latest_exp}/checkpoint/best"
    
    # Analyze
    analyze_cmd = [
        "python", "scripts/run_phase4_analyze.py",
        "--checkpoint", checkpoint_path,
        "--split", "validation",
        "--data-config", "configs/mac_data.yaml",
        "--dataset-version", str(version)
    ]
    subprocess.run(analyze_cmd, check=True)
    print(f"Pipeline for dataset_v{version} complete!")


def main():
    sweep = [(20, 2), (40, 3), (60, 4)]
    
    for n_samples, version in sweep:
        create_partial_curriculum(n_samples, version)
        run_pipeline(version)
        
    print("\nExperiment 1 (Curriculum Sweep) finished successfully.")


if __name__ == "__main__":
    main()
