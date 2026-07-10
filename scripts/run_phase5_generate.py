"""
scripts/run_phase5_generate.py

Phase 5: Adaptive Hard-Negative Generation
───────────────────────────────────────────
Select student failures, generate targeted hard-negative examples
using the teacher, verify them, and merge into a new dataset version.

NOTE: This phase requires implementing the research methodology in:
    - src/adaptive/failure_selection.py
    - src/adaptive/hard_negative_generation.py

Until those are implemented, this script will raise NotImplementedError.

Usage:
    python scripts/run_phase5_generate.py --failure-report experiments/exp002/failure_report.json
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Adaptive Hard-Negative Generation"
    )
    parser.add_argument(
        "--failure-report", type=Path, required=True,
        help="Path to failure report JSON from Phase 4.",
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--teacher-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "teacher.yaml",
    )
    parser.add_argument(
        "--adaptive-config", type=Path,
        default=_PROJECT_ROOT / "configs" / "adaptive.yaml",
    )
    parser.add_argument(
        "--dataset-version", type=int, default=0,
        help="Source dataset version to augment.",
    )
    args = parser.parse_args()

    setup_logging(log_file=_PROJECT_ROOT / "logs" / "phase5.log")

    logger.info("━" * 60)
    logger.info("PHASE 5: Adaptive Hard-Negative Generation")
    logger.info("━" * 60)

    # 1. Setup Backend
    teacher_config = load_config(args.teacher_config)
    adaptive_config = load_config(args.adaptive_config)
    data_config = load_config(args.data_config)
    
    from src.teacher.backend import create_backend
    backend_kwargs = teacher_config.get("backend_kwargs", {})
    backend = create_backend(
        backend_type=teacher_config.get("backend", "transformers"),
        model_id=teacher_config["model_id"],
        **backend_kwargs
    )
    
    # 2. Paths
    failure_report_path = args.failure_report
    failures_jsonl_path = failure_report_path.parent / "failures.jsonl"
    
    experiments_dir = _PROJECT_ROOT / teacher_config.get("experiments_dir", "experiments")
    from src.utils.experiment import init_experiment, get_dataset_version_dir
    exp_dir = init_experiment(
        experiments_dir, 
        config={"adaptive": adaptive_config, "teacher": teacher_config},
        phase="phase5_generate",
        description=f"Curriculum Generation for v{args.dataset_version}"
    )
    
    processed_dir = _PROJECT_ROOT / data_config.get("processed_dir", "data/processed")
    source_dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version)
    target_dataset_dir = get_dataset_version_dir(processed_dir, args.dataset_version + 1)
    
    # 3. Teacher Diagnosis
    from src.adaptive.teacher_diagnosis import TeacherDiagnostician
    diagnostician = TeacherDiagnostician(output_dir=exp_dir, teacher_client=backend)
    diagnostician.generate_diagnosis(failure_report_path, failures_jsonl_path)
    
    # Load source train to sample benign contexts
    import pandas as pd
    source_train = pd.read_csv(source_dataset_dir / "train.csv")
    benign_contexts_df = source_train[source_train["label"] == 0].copy()
    if len(benign_contexts_df) == 0:
        logger.error("No benign contexts found in source dataset to sample from.")
        sys.exit(1)
        
    # Extract config
    gen_config = adaptive_config.get("generation", {})
    total_samples = gen_config.get("total_curriculum_samples", 100)
    allocation_strategy = gen_config.get("allocation_strategy", "proportional")
    top_k_multiplier = gen_config.get("top_k_multiplier", 5)
    benign_ratio = gen_config.get("benign_ratio", 0.5)

    # Read all failures and partition by attack_family
    import json
    from collections import defaultdict
    import math

    clusters = defaultdict(list)
    if failures_jsonl_path.exists():
        with open(failures_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                fail_data = json.loads(line)
                fail_text = fail_data.get("text", fail_data.get("context", ""))
                fam = fail_data.get("attack_family", "unknown")
                if fail_text:
                    clusters[fam].append(fail_data)
                    
    total_failures = sum(len(c) for c in clusters.values())
    if total_failures == 0:
        logger.info("No valid failures found. Skipping curriculum generation.")
        mapped_targets = []
    else:
        # Calculate allocations
        cluster_allocations = {}
        if allocation_strategy == "proportional":
            for fam, fails in clusters.items():
                cluster_allocations[fam] = max(1, math.floor((len(fails) / total_failures) * total_samples))
        else:
            # Fallback to equal
            per_cluster = max(1, total_samples // len(clusters))
            for fam in clusters.keys():
                cluster_allocations[fam] = per_cluster

        # Partitioned Semantic Filtering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        mapped_targets = []
        
        benign_texts = benign_contexts_df["context"].tolist()
        benign_ids = benign_contexts_df["id"].tolist()

        for fam, fails in clusters.items():
            n_alloc = cluster_allocations[fam]
            logger.info(f"Processing cluster '{fam}' ({len(fails)} failures) -> Allocating {n_alloc} curriculum samples.")
            
            failure_texts = [f.get("text", f.get("context", "")) for f in fails]
            
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            try:
                tfidf_benign = vectorizer.fit_transform(benign_texts)
                tfidf_failures = vectorizer.transform(failure_texts)
            except ValueError:
                logger.warning(f"TF-IDF failed for cluster '{fam}'. Skipping.")
                continue
                
            sim_matrix = cosine_similarity(tfidf_benign, tfidf_failures)
            
            # Max similarity to any failure in this cluster
            max_sims = sim_matrix.max(axis=1)
            # The index of the failure that gave the max similarity
            best_fail_indices = sim_matrix.argmax(axis=1)
            
            # Create a dataframe to sort and sample
            cluster_df = pd.DataFrame({
                "context": benign_texts,
                "context_id": benign_ids,
                "similarity": max_sims,
                "best_fail_idx": best_fail_indices
            })
            
            top_k = max(n_alloc * top_k_multiplier, 10)
            cluster_top_k = cluster_df.sort_values("similarity", ascending=False).head(top_k)
            
            # Sample without replacement
            if len(cluster_top_k) >= n_alloc:
                sampled = cluster_top_k.sample(n_alloc, replace=False)
            else:
                logger.warning(f"Not enough top-K contexts for '{fam}'. Using replace=True.")
                sampled = cluster_top_k.sample(n_alloc, replace=True)
                
            for _, row in sampled.iterrows():
                fail_record = fails[row["best_fail_idx"]]
                mapped_targets.append({
                    "originating_failure_id": fail_record.get("id", "unknown"),
                    "retrieved_context_id": row["context_id"],
                    "context": row["context"],
                    "attack_family": fam
                })
                
        logger.info(f"Total mapped curriculum targets: {len(mapped_targets)}")

    # 4. Curriculum Generation
    from src.adaptive.curriculum_generation import CurriculumGenerator
    generator = CurriculumGenerator(output_dir=exp_dir, teacher_client=backend)
    
    diagnosis_path = exp_dir / "diagnosis.json"
        
    generator.generate_curriculum(
        diagnosis_path, 
        mapped_targets=mapped_targets,
        benign_ratio=benign_ratio,
        dataset_version=args.dataset_version
    )
    
    # 5. Merging & Versioning
    import shutil
    
    curriculum_path = exp_dir / "generated_curriculum.jsonl"
    
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Validation and Test
    for split in ["validation.csv", "test.csv"]:
        src_split = source_dataset_dir / split
        if src_split.exists():
            shutil.copy(src_split, target_dataset_dir / split)
            
    # Merge Train
    source_train = pd.read_csv(source_dataset_dir / "train.csv")
    
    curr_data = []
    with open(curriculum_path, "r", encoding="utf-8") as f:
        for line in f:
            curr_data.append(json.loads(line))
            
    if curr_data:
        curr_df = pd.DataFrame(curr_data)
        # Ensure ID format
        curr_df["id"] = [f"curr_{i}" for i in range(len(curr_df))]
        target_train = pd.concat([source_train, curr_df], ignore_index=True)
    else:
        target_train = source_train
        
    target_train.to_csv(target_dataset_dir / "train.csv", index=False)
    
    logger.info(f"Phase 5 complete. New dataset version {args.dataset_version + 1} saved to {target_dataset_dir}")
    logger.info(f"curriculum saved to {curriculum_path}")


if __name__ == "__main__":
    main()
