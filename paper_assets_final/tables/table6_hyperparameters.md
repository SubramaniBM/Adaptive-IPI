# Table VI — Implementation Hyperparameters

| Hyperparameter | Value |
|:---|:---|
| Student Model | answerdotai/ModernBERT-base |
| Teacher Model | Qwen (frozen) |
| Optimizer | AdamW |
| Learning Rate | 2.0 × 10⁻⁵ |
| Weight Decay | 0.01 |
| Batch Size (per device) | 8 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 64 |
| Max Sequence Length | 512 |
| Max Epochs | 30 |
| Early Stopping Patience | 3 |
| Monitor Metric | F1 |
| KD Temperature (τ) | 4.0 |
| KD Alpha (α) | 0.7 |
| Warmup Ratio | 0.1 |
| Max Grad Norm | 1.0 |
| Backend | Apple Silicon MPS |
| Framework | PyTorch + HuggingFace Transformers |
