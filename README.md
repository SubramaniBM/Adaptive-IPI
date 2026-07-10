# Adaptive-IPI

**Adaptive Teacher-Guided Hard-Negative Curriculum for Lightweight Indirect Prompt Injection Detection**

> Academic research implementation targeting ACL/EMNLP publication.

## Research Objective

Develop a lightweight detector for Embedded Instruction Insertion (Indirect Prompt Injection) attacks in retrieved documents. The deployed detector is CPU-friendly (ModernBERT-base) while maintaining strong robustness against unseen attacks.

**Novelty**: An adaptive teacher-guided hard-negative curriculum that improves a lightweight detector by generating targeted examples based on the student's observed weaknesses.

## Pipeline

| Phase | Description | Script |
|-------|-------------|--------|
| 1 | Dataset preprocessing (BIPIA) | `scripts/run_phase1_preprocess.py` |
| 2 | Teacher annotation (Qwen3-32B) | `scripts/run_phase2_teacher.py` |
| 3 | Initial ModernBERT knowledge distillation | `scripts/run_phase3_distill.py` |
| 4 | Validation and failure analysis | `scripts/run_phase4_analyze.py` |
| 5 | Adaptive hard-negative generation | `scripts/run_phase5_generate.py` |
| 6 | Retraining with augmented data | `scripts/run_phase6_retrain.py` |
| 7 | Final evaluation | `scripts/run_phase7_evaluate.py` |

Only the final ModernBERT model is deployed. The teacher is **never** used during inference.

## Models

| Role | Model |
|------|-------|
| Teacher (offline) | Qwen3-32B-Instruct |
| Student (deployed) | ModernBERT-base |

## Setup

```bash
# Clone and enter project
cd Adaptive-IPI

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone and install BIPIA
git clone https://github.com/microsoft/BIPIA.git data/raw/bipia
pip install -e data/raw/bipia
```

## Quick Start

```bash
# Phase 1: Preprocess BIPIA dataset
python scripts/run_phase1_preprocess.py

# Phase 3: Train baseline (hard labels only)
python scripts/run_phase3_distill.py --loss-type ce
```

## Project Structure

```
Adaptive-IPI/
├── configs/           # YAML configuration files
├── data/              # Dataset files (versioned, immutable)
├── src/
│   ├── core/          # Types, constants, enums
│   ├── datasets/      # Data loading and preprocessing
│   ├── teacher/       # Teacher annotation pipeline
│   ├── models/        # Student model definition
│   ├── training/      # Custom training engine and losses
│   ├── adaptive/      # Hard-negative curriculum pipeline
│   ├── evaluation/    # Metrics, calibration, analysis
│   └── utils/         # Configuration, logging, I/O
├── scripts/           # Phase CLI entry points
├── experiments/       # Experiment artifacts (auto-numbered)
├── checkpoints/       # Model checkpoints
├── logs/              # Training logs
├── figures/           # Generated plots
└── notebooks/         # Jupyter notebooks
```

## Tech Stack

- Python 3.11
- PyTorch
- Transformers (HuggingFace)
- Datasets
- scikit-learn
- Pandas / NumPy

## Key Design Decisions

1. **Custom training engine** (not HF Trainer) for full control over CE/KD/adaptive losses
2. **Separated teacher backend** from annotation logic (swap Qwen → Gemini without touching annotation code)
3. **Versioned datasets** (`dataset_v0/`, `dataset_v1/`, ...) — every dataset is immutable
4. **Full experiment saving** — config, metrics, predictions, plots per experiment
5. **Calibration metrics** — ECE, reliability diagrams, confidence histograms
6. **Rich teacher annotations** — probs, entropy, reasoning stored for post-hoc analysis
7. **No placeholder methodology** — research decisions raise `NotImplementedError` until specified
