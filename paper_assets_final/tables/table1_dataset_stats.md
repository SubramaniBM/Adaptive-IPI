# Table I — Dataset Statistics

| Split | Source | Benign | Attack | Total |
|:---|:---|---:|---:|---:|
| Train (V4) | BIPIA + Enron Augmentation | 1,386 | 1,401 | 2,787 |
| Validation | BIPIA | 88 | 1,500 | 1,588 |
| Test (In-Domain) | BIPIA | 344 | 7,575 | 7,919 |
| OOD — Handcrafted Benign | Hand-crafted | 300 | 0 | 300 |
| OOD — Enron Held-out | Enron corpus (deduplicated) | 925 | 0 | 925 |

**Notes:**
- V4 training set augments the V1 BIPIA training data with 1,000 curated Enron enterprise emails.
- Enron Held-out emails with >85% TF-IDF cosine similarity to training emails were removed, leaving 925 disjoint emails.
- Train V1 (Baseline): 386 benign + 1,401 attack = 1,787 total.
