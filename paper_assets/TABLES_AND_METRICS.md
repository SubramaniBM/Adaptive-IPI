# Core Paper Metrics & Tables

## 1. 4-Seed Reproducibility Statistics (V1 Curriculum)

| Run | Seed | Precision | Recall | F1 Score | AUROC | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1** | `42` | `0.9941` | `0.9482` | `0.9707` | `0.9542` | `0.1571` |
| **Run 2** | `43` | `0.9630` | `0.9935` | `0.9780` | `0.9545` | `0.1030` |
| **Run 3** | `44` | `0.9715` | `0.9936` | `0.9824` | `0.9610` | `0.1130` |
| **Run 4** | `45` | `0.9883` | `0.9720` | `0.9801` | `0.9569` | `0.1378` |

**Statistical Summary**
* **Accuracy:** `95.78% ± 0.80%`
* **Precision:** `0.9831 ± 0.0119`
* **Recall:** `0.9729 ± 0.0162`
* **F1 Score:** `0.9778 ± 0.0044`
* **AUROC:** `0.9632 ± 0.0077`
* **AUPRC:** `0.9980 ± 0.0005`
* **Expected Calibration Error (ECE):** `0.1259 ± 0.0226`
* **Final Reported F1 Metric:** **`97.78% ± 0.44%`**

---

## 2. Baseline Comparison (V0 vs V1)

| Metric | Original Broken V1 | V0 Baseline (Aligned) | V1 Curriculum (Aligned) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 17.44% | 97.80% | **99.06%** 🏆 |
| **Precision** | 100.0% | 97.84% | **99.14%** 🏆 |
| **Recall** | 12.60% | 99.87% | **99.87%** 🏆 |
| **F1 Score** | 0.2238 | 0.9884 | **0.9950** 🏆 |
| **AUROC** | 0.8220 | 0.9934 | **0.9978** 🏆 |
| **AUPRC** | 0.9880 | 0.9996 | **0.9999** 🏆 |
| **ECE (Calib)**| 0.6704 | 0.1169 | **0.1096** 🏆 *(Lower is better)* |

---

## 3. Confusion Matrix Breakdown (Seed 43)

* **True Negatives (Correctly identified Benign):** 55
* **True Positives (Correctly identified Attacks):** 7,526
* **False Positives (Mistakenly flagged Benign as Attack):** 289
* **False Negatives (Missed an actual Attack):** 49
