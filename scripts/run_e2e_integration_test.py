import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("Starting E2E Integration Test...")
    start_time = time.time()
    report = []
    report.append("# End-to-End Integration Test Report\n")
    
    # 1. Prepare Dataset (dataset_v99)
    src_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v0"
    dst_dir = _PROJECT_ROOT / "data" / "processed" / "dataset_v99"
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and subsample
    train_df = pd.read_csv(src_dir / "train.csv").sample(50, random_state=42)
    val_df = pd.read_csv(src_dir / "validation.csv").sample(20, random_state=42)
    test_df = pd.read_csv(src_dir / "test.csv").sample(20, random_state=42)
    
    train_df.to_csv(dst_dir / "train.csv", index=False)
    val_df.to_csv(dst_dir / "validation.csv", index=False)
    test_df.to_csv(dst_dir / "test.csv", index=False)
    
    print(f"Created subsampled dataset at {dst_dir}")
    
    # 2. Configs
    teacher_cfg = _PROJECT_ROOT / "configs" / "e2e_teacher.yaml"
    student_cfg = _PROJECT_ROOT / "configs" / "e2e_student.yaml"
    data_cfg = _PROJECT_ROOT / "configs" / "e2e_data.yaml"
    
    with open(teacher_cfg, "w") as f:
        f.write("model_id: 'Qwen/Qwen2.5-0.5B-Instruct'\n")
        f.write("backend: 'transformers'\n")
        f.write("batch_size: 1\n")
        f.write("backend_kwargs: {}\n")
        f.write("annotations_path: 'data/processed/dataset_v99/teacher_annotations.jsonl'\n")
        
    with open(student_cfg, "w") as f:
        f.write("model_name: 'answerdotai/ModernBERT-base'\n")
        f.write("num_epochs: 1\n")
        f.write("batch_size: 2\n")
        f.write("learning_rate: 0.00002\n")
        f.write("max_length: 512\n")
        f.write("experiments_dir: 'outputs/e2e_experiments'\n")
        
    with open(data_cfg, "w") as f:
        f.write("processed_dir: 'data/processed'\n")
        f.write("teacher_annotations_path: 'data/processed/dataset_v99/teacher_annotations.jsonl'\n")
        f.write("seed: 42\n")
        
    def run_stage(name, cmd):
        print(f"\n--- Running Stage: {name} ---")
        t0 = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        t1 = time.time()
        dur = t1 - t0
        success = result.returncode == 0
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        report.append(f"## {name}")
        report.append(f"- **Status**: {status}")
        report.append(f"- **Execution Time**: {dur:.2f} seconds")
        
        if not success:
            report.append(f"\n**Error Output:**\n```\n{result.stderr[:2000]}\n{result.stdout[:2000]}\n```")
            print(f"Stage {name} failed!")
            print(result.stderr)
            print(result.stdout)
            return False, dur, result
        return True, dur, result
        
    # Stages
    stages = [
        ("Teacher Annotation", f"python3 scripts/run_phase2_teacher.py --dataset-version 99 --config {teacher_cfg} --data-config {data_cfg}"),
        ("Knowledge Distillation", f"python3 scripts/run_phase3_distill.py --dataset-version 99 --student-config {student_cfg} --data-config {data_cfg} --loss-type kd"),
    ]
    
    all_success = True
    total_time = 0
    phase3_exp_dir = None
    
    for name, cmd in stages:
        success, dur, res = run_stage(name, cmd)
        total_time += dur
        if not success:
            all_success = False
            break
            
        if name == "Knowledge Distillation":
            # extract exp dir from logs
            for line in res.stdout.split("\n") + res.stderr.split("\n"):
                if "Experiment:" in line:
                    phase3_exp_dir = line.split("Experiment:")[1].strip()
                    print(f"Found Phase 3 Exp Dir: {phase3_exp_dir}")
                    
    if all_success and phase3_exp_dir:
        # Phase 4
        success, dur, res = run_stage("Failure Profile (Evaluation)", f"python3 scripts/run_phase4_analyze.py --dataset-version 99 --checkpoint {phase3_exp_dir}/checkpoint/best --data-config {data_cfg}")
        total_time += dur
        if success:
            phase4_exp_dir = None
            for line in res.stdout.split("\n") + res.stderr.split("\n"):
                if "Report:" in line:
                    phase4_exp_dir = line.split("Report:")[1].strip()
            
            if phase4_exp_dir:
                failure_report_path = f"{phase4_exp_dir}/failure_report.json"
                
                # Phase 5
                success, dur, res = run_stage("Teacher Diagnosis & Curriculum", f"python3 scripts/run_phase5_generate.py --dataset-version 99 --failure-report {failure_report_path} --data-config {data_cfg} --teacher-config {teacher_cfg}")
                total_time += dur
                
                if success:
                    curriculum_path = None
                    for line in res.stdout.split("\n") + res.stderr.split("\n"):
                        if "curriculum saved to" in line.lower():
                            curriculum_path = line.split("to")[-1].strip()
                    
                    if not curriculum_path:
                        # Fallback
                        curriculum_path = "data/generated/curriculum.jsonl"
                        
                    # Phase 6
                    success, dur, res = run_stage("Retraining", f"python3 scripts/run_phase6_retrain.py --dataset-version 99 --student-config {student_cfg} --data-config {data_cfg}")
                    total_time += dur
                    
                    if success:
                        phase6_exp_dir = None
                        for line in res.stdout.split("\n") + res.stderr.split("\n"):
                            if "Experiment:" in line:
                                phase6_exp_dir = line.split("Experiment:")[1].strip()
                                
                        if phase6_exp_dir:
                            # Phase 7
                            success, dur, res = run_stage("Final Evaluation", f"python3 scripts/run_phase7_evaluate.py --dataset-version 99 --checkpoint {phase6_exp_dir}/checkpoint/best --data-config {data_cfg}")
                            total_time += dur
                        else:
                            all_success = False
                    else:
                        all_success = False
                else:
                    all_success = False
            else:
                all_success = False
        else:
            all_success = False

    report.insert(1, f"**Total Execution Time**: {total_time:.2f} seconds\n")
    report.insert(2, f"**Overall Status**: {'✅ SUCCESS' if all_success else '❌ FAILED'}\n")
    
    report_path = _PROJECT_ROOT / "reports" / "end_to_end_smoke_test.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"\nIntegration test complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
