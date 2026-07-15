import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets.dataset import IPIDataset
from src.models.student import create_student
from src.training.losses import DistillationLoss

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = create_student("answerdotai/ModernBERT-base")
    model.to(device)
    
    dataset = IPIDataset(
        data_path=_PROJECT_ROOT / "data/processed/dataset_v2/train.csv",
        tokenizer=tokenizer,
        max_length=512,
        teacher_annotations_path=_PROJECT_ROOT / "data/processed/teacher_annotations.jsonl"
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)
    
    print("Testing forward passes to isolate NaN...")
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        t_probs = batch["teacher_probs"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if torch.isnan(logits).any():
                print(f"Batch {i} generated NaN logits!")
                return
                
            loss = loss_fn(student_logits=logits, labels=labels, teacher_probs=t_probs)
            if torch.isnan(loss):
                print(f"Batch {i} generated NaN loss!")
                return
                
    print("No NaNs detected in forward pass. Must be an optimizer/LR issue.")

if __name__ == "__main__":
    main()
