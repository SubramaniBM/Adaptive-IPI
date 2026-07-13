import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset import IPIDataset
from src.models.student import create_student

def main():
    model, tokenizer = create_student("answerdotai/ModernBERT-base")
    dataset = IPIDataset(
        data_path=_PROJECT_ROOT / "data/processed/dataset_v1/train.csv",
        tokenizer=tokenizer,
        max_length=512,
        teacher_annotations_path=_PROJECT_ROOT / "data/processed/teacher_annotations.jsonl"
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        t_probs = batch.get("teacher_probs")
        if t_probs is not None:
            if torch.isnan(t_probs).any():
                print(f"Batch {i} has NaN in teacher_probs!")
                print(t_probs)
            
            # Check if any probability is outside [0, 1]
            if (t_probs < 0).any() or (t_probs > 1).any():
                print(f"Batch {i} has out of bounds teacher_probs!")
                print(t_probs)
                
            # Simulate what happens in KD loss
            t_logits = torch.log(t_probs.clamp(min=1e-8))
            if torch.isnan(t_logits).any():
                print(f"Batch {i} has NaN in teacher_logits!")
                print(t_logits)
                
            soft_t = F.softmax(t_logits / 4.0, dim=-1)
            if torch.isnan(soft_t).any():
                print(f"Batch {i} has NaN in soft_teacher!")
                
    print("Debug complete.")

if __name__ == "__main__":
    main()
