# Dataset Construction Process

This document explains how the Adaptive-IPI binary classification dataset is built from the official BIPIA benchmark resources.

## BIPIA Organization
The official BIPIA dataset is decoupled:
1. **Benign Contexts:** Host text (e.g., normal emails) stored in `benchmark/{task}/{split}.jsonl`.
2. **Attack Library:** Malicious payloads stored in `benchmark/text_attack_{split}.json` (and `code_attack_{split}.json`).

## Why AutoPIABuilder is Bypassed
BIPIA provides an `AutoPIABuilder` to generate datasets. However, we bypass it for the following reasons:
1. **No Benign Samples:** The builder is purely a combinatorial poisoning engine. It only yields attacks and drops all clean contexts.
2. **ASR Optimization:** The builder formulates the dataset for Attack Success Rate (ASR) instruction-following evaluation, not binary classification.

## Our DatasetBuilder Pipeline
To formulate a binary classification task while remaining faithful to the benchmark, our pipeline works as follows:

1. **`BIPIALoader`**: Solely responsible for loading the raw JSONL contexts and JSON attack payloads into memory as Python objects.
2. **`DatasetBuilder`**: The central engine.
    - Yields the benign contexts unaltered, labeling them `0` (Benign).
    - Uses combinatorial expansion (`contexts` × `attacks` × `positions`) to create attack samples, labeling them `1` (Attack).
3. **`injection.py`**: A wrapper module that strictly imports and executes BIPIA's official `insert_start`, `insert_middle`, and `insert_end` utility functions. This ensures our dataset's string manipulation is 100% faithful to the official benchmark.

## Metadata Preservation
Every sample strictly preserves:
- The unmodified BIPIA `context` string (or the successfully injected poisoned string).
- The raw `attack_instruction` payload.
- `attack_family` and `attack_position`.
- Task origin (e.g., `email`).

## Reproducibility Guarantees
- The `DatasetBuilder` generates a deterministic `dataset_statistics.json` alongside the data.
- The statistics file embeds reproducibility metadata (`construction_version`, `seed`, `benchmark_version`, `generated_on`).
- A `construction_config.yaml` is also saved alongside the processed data so the exact dataset can be rebuilt flawlessly.
