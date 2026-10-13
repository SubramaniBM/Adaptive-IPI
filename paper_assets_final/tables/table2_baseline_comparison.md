# Table II — Comparison of Progressive Training Strategies

| Configuration | Description | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | ECE | HC FPR | Enron FPR |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 — Static KD | KD on BIPIA, no adaptive refinement, no Enron | 97.12% | 99.28% | 97.70% | 98.48% | 96.69% | 99.78% | 0.1169 | 44.00% | 98.81% |
| V1 — Adaptive-IPI | V0 + adaptive hard-negative curriculum | 94.52% | 99.41% | 94.84% | 97.07% | 97.32% | 99.87% | 0.1096 | 44.00% | 98.81% |
| Enron250 | V1 + 250 Enron emails | 96.21% | 99.56% | 96.46% | 97.99% | 98.29% | 99.92% | 0.1486 | 38.00% | 0.1081% |
| Enron500 | V1 + 500 Enron emails | 96.49% | 99.55% | 96.77% | 98.14% | 98.92% | 99.95% | 0.1217 | 58.33% | 0.00% |
| **Enron1000** | V1 + 1000 Enron emails | **96.54%** | **99.57%** | **96.81%** | **98.17%** | **98.01%** | **99.91%** | 0.1193 | 36.67% | **0.00%** |

Checkpoints:
- V0 Static KD: outputs/mac_experiments/exp008/checkpoint/best
- V1 Adaptive-IPI: outputs/mac_experiments/exp009/checkpoint/best
- Enron250: experiments/exp048/checkpoint/best
- Enron500: experiments/exp049/checkpoint/best
- Enron1000: experiments/exp050/checkpoint/best (seed 42)
