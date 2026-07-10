# GPU Readiness Report

**Generated**: 2026-07-10T08:56 IST  
**Repository**: Adaptive-IPI  
**Status**: ✅ READY FOR GPU EXPERIMENTS

---

## Checklist

### Dataset
| Check | Status |
| :--- | :--- |
| Dataset frozen (v0) | ✅ |
| Zero context overlap (Train/Val/Test) | ✅ |
| Zero duplicate IDs | ✅ |
| Zero duplicate (context, intent) pairs | ✅ |
| Labels binary (0/1) | ✅ |
| 29 attack families covered | ✅ |
| 3 attack positions covered (start/middle/end) | ✅ |
| Integrity report | ✅ `reports/dataset_integrity_report.json` |

### Environment
| Check | Status |
| :--- | :--- |
| `requirements.txt` (curated) | ✅ 29 packages |
| `requirements_frozen.txt` (full freeze) | ✅ 179 packages |
| All configs audited | ✅ |

### Configuration Audit
| Parameter | Config File | Status |
| :--- | :--- | :--- |
| learning_rate | `configs/student.yaml` | ✅ `2.0e-5` |
| batch_size | `configs/student.yaml` | ✅ `16` |
| num_epochs | `configs/student.yaml` | ✅ `5` |
| KD temperature | `configs/distillation.yaml` | ✅ `4.0` |
| KD alpha | `configs/distillation.yaml` | ✅ `0.7` |
| random seed | `configs/data.yaml` | ✅ `42` |
| teacher model | `configs/teacher.yaml` | ✅ `Qwen/Qwen3-32B-Instruct` |
| student model | `configs/student.yaml` | ✅ `answerdotai/ModernBERT-base` |
| dataset paths | `configs/data.yaml` | ✅ `data/processed` |
| checkpoint paths | `configs/student.yaml` | ✅ `checkpoints` |

### CPU Smoke Test
| Check | Status |
| :--- | :--- |
| Dataset loads | ✅ |
| Tokenizer loads | ✅ |
| ModernBERT forward pass | ✅ Logits shape: (B, 2) |
| CE Loss computes | ✅ 0.5278 |
| Backpropagation executes | ✅ Gradients verified |
| Checkpoint saving | ✅ `config.json` confirmed |
| Evaluation loop executes | ✅ F1, accuracy, AUROC, ECE |
| Report | ✅ `reports/cpu_smoke_test.md` |

### Teacher Dry Run
| Check | Status |
| :--- | :--- |
| Real pipeline used (no mocks) | ✅ |
| Model loaded (Qwen2.5-0.5B-Instruct) | ✅ |
| 5/5 samples annotated | ✅ |
| Predictions present (0/1) | ✅ |
| Probabilities present ([p₀, p₁]) | ✅ |
| Reasoning present | ✅ |
| JSON format valid | ✅ |
| Report | ✅ `reports/teacher_dry_run.json` |

### KD Dry Run
| Check | Status |
| :--- | :--- |
| Consumed REAL teacher outputs | ✅ `teacher_dry_run.jsonl` |
| Teacher probs merged into dataset | ✅ 5/5 matched |
| CE loss computed | ✅ 0.773 |
| KD loss computed | ✅ 3.500 |
| Combined loss backpropagated | ✅ |
| Gradients computed | ✅ |
| Report | ✅ `reports/kd_dry_run.json` |

### Logging
| Check | Status |
| :--- | :--- |
| `MetricLoggerCallback` logs to JSONL | ✅ `training_log.jsonl` |
| Console logs: train_loss, lr, eval metrics | ✅ |
| `BestModelCallback` tracks best F1 | ✅ |
| `EarlyStoppingCallback` monitors F1 | ✅ |
| Experiment metadata saved | ✅ `metadata.json` per experiment |

### Output Directories
| Directory | Status |
| :--- | :--- |
| `outputs/checkpoints/` | ✅ |
| `outputs/teacher/` | ✅ |
| `outputs/logs/` | ✅ |
| `outputs/failure_profiles/` | ✅ |
| `outputs/curriculum/` | ✅ |
| `outputs/evaluation/` | ✅ |

---

## Bugs Fixed During Preparation

1. **Missing `src/adaptive/failure_analysis.py`** — `evaluate_model()` imported `run_inference` from a module that didn't exist. Created the module.
2. **Column mismatch** — `IPIDataset` expected a `text` column but the frozen CSV uses `context`. Added an automatic rename bridge.
3. **`pd.notna()` on list values** — Teacher probs stored as lists caused ambiguous truth value errors in `__getitem__`. Fixed the null check.
4. **vLLM kwargs leaking to Transformers** — `tensor_parallel_size` from `teacher.yaml` was passed to `AutoModelForCausalLM`. Cleared backend-specific kwargs in dry run.
5. **`torch_dtype` deprecation** — Updated to `dtype` parameter in `TransformersBackend`.

---

## Verdict

**✅ Repository is GPU-ready.**

The next step is to run the full pipeline on GPU:

```
Teacher Annotation → KD → Failure Profile → Curriculum → Retraining
```

No further infrastructure changes needed.
