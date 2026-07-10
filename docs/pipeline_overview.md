# Pipeline Overview

## Data Flow

```
BIPIA (raw)
    │
    ▼
Phase 1: Dataset Construction & Preprocessing
    │ a) BIPIALoader (read raw JSON/JSONL)
    │ b) DatasetBuilder (inject attacks, label samples, output stats)
    │ c) Preprocessing (clean text, deduplicate, validate schema)
    │ → data/processed/dataset_v0/
    │
    ▼
Phase 2: Teacher Annotation (Qwen3-32B)
    │ → data/processed/teacher_annotations.jsonl
    │   Fields: id, teacher_prediction, teacher_probs,
    │           teacher_entropy, teacher_reasoning
    │
    ▼
Phase 3: Initial KD Training
    │ → experiments/exp001/ (config, metrics, checkpoint)
    │ → Student model (ModernBERT-base)
    │
    ▼
Phase 4: Failure Analysis
    │ → experiments/exp002/ (failure_report.json, failures.csv)
    │   Categorised by: failure_type × attack_type × context_type
    │
    ▼
Phase 5: Adaptive Hard-Negative Generation [RESEARCH NOVELTY]
    │ a) Failure selection (which failures to target?)
    │ b) Teacher-guided generation (create targeted examples)
    │ c) Verification (quality checks)
    │ → data/generated/
    │ → data/processed/dataset_v1/ (augmented, immutable)
    │
    ▼
Phase 6: Retraining
    │ → experiments/exp003/ (new checkpoint, metrics)
    │
    ▼
Phase 7: Evaluation
    │ → experiments/exp004/
    │   - metrics.json
    │   - predictions.csv
    │   - confusion_matrix.png
    │   - reliability_diagram.png
    │   - confidence_histogram.png
    │   - evaluation_report.txt
    │
    ▼
Compare: Pre-curriculum vs. Post-curriculum
    → figures/comparison.png
```

## Key Invariants

1. **Raw data is never modified** (`data/raw/` is read-only)
2. **Datasets are versioned and immutable** (`dataset_v0/`, `dataset_v1/`, ...)
3. **Experiments are never overwritten** (`exp001/`, `exp002/`, ...)
4. **Teacher is never used at inference time** (student-only deployment)
5. **Research decisions are not implemented by the coding agent**
   (they raise `NotImplementedError`)
