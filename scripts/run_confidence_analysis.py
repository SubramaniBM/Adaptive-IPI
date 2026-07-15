import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import json

def main():
    model_path = "outputs/mac_experiments/exp009/checkpoint/best"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    df = pd.read_csv("data/processed/handcrafted_benign.csv")
    
    predictions = []
    attack_probs = []
    
    print("Running inference...")
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        attack_p = probs[:, 1].cpu().numpy()
        
        predictions.extend(preds)
        attack_probs.extend(attack_p)
        
    df["predicted_label"] = predictions
    df["attack_prob"] = attack_probs
    
    false_positives = df[df["predicted_label"] == 1]
    fp_rate = len(false_positives) / len(df)
    
    print(f"\nHandcrafted Benign False Positive Rate: {fp_rate:.2%}")
    
    # Confidence analysis
    high_conf_fp = false_positives[false_positives["attack_prob"] > 0.90]
    borderline_fp = false_positives[(false_positives["attack_prob"] >= 0.50) & (false_positives["attack_prob"] <= 0.60)]
    
    stats = {
        "total_handcrafted": len(df),
        "false_positives": len(false_positives),
        "fp_rate": float(fp_rate),
        "high_confidence_fp_pct": float(len(high_conf_fp) / max(1, len(false_positives)) * 100),
        "borderline_fp_pct": float(len(borderline_fp) / max(1, len(false_positives)) * 100),
        "avg_fp_confidence": float(false_positives["attack_prob"].mean()) if len(false_positives) > 0 else 0.0
    }
    
    print("\n--- CONFIDENCE ANALYSIS ---")
    print(f"Total FP: {len(false_positives)}")
    print(f"Avg FP Attack Probability: {stats['avg_fp_confidence']:.4f}")
    print(f"High Confidence (>90%): {stats['high_confidence_fp_pct']:.2f}%")
    print(f"Borderline (50-60%): {stats['borderline_fp_pct']:.2f}%")
    
    os.makedirs("reports/distribution_analysis", exist_ok=True)
    with open("reports/distribution_analysis/confidence_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        
    # Save the false positives for failure taxonomy later
    false_positives.to_csv("reports/distribution_analysis/handcrafted_false_positives.csv", index=False)

if __name__ == "__main__":
    main()
