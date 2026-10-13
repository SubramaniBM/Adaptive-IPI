# Table IV — Benign Distribution Expansion

| Configuration | BIPIA Accuracy | BIPIA F1 | BIPIA AUROC | Handcrafted FPR | Enron Held-out FPR |
|:---|---:|---:|---:|---:|---:|
| Baseline (BIPIA only) | 94.52% | 97.07% | 97.32% | 44.00% | 98.81% |
| Enron250 | 96.21% | 97.99% | 98.29% | 38.00% | 0.1081% |
| Enron500 | 96.49% | 98.14% | 98.92% | 58.33% | 0.00% |
| **Enron1000** | **96.54%** | **98.17%** | **98.01%** | **36.67%** | **0.00%** |

**Evaluation set:** Deduplicated Enron Held-out (925 disjoint emails, >85% TF-IDF similarity threshold applied).
