import os
import glob
import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def evaluate(model, tokenizer, device, df, is_ood=False):
    texts = (df['context'] if 'context' in df.columns else df['text']).tolist()
    labels = df['label'].tolist() if 'label' in df.columns else [0] * len(texts)
    
    all_preds = []
    all_probs = []
    
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), leave=False):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
        
    all_probs = np.array(all_probs)
    attack_probs = all_probs[:, 1]
    
    if is_ood:
        # All OOD sets are benign (0)
        false_positives = np.sum(np.array(all_preds) == 1)
        fpr = float(false_positives / len(texts))
        avg_conf = float(np.mean(attack_probs[np.array(all_preds) == 1])) if false_positives > 0 else 0.0
        return {
            "false_positive_rate": fpr,
            "avg_fp_confidence": avg_conf
        }
    else:
        # Standard metrics
        return {
            "accuracy": accuracy_score(labels, all_preds),
            "precision": precision_score(labels, all_preds, zero_division=0),
            "recall": recall_score(labels, all_preds, zero_division=0),
            "f1": f1_score(labels, all_preds, zero_division=0),
            "auroc": roc_auc_score(labels, attack_probs),
            "auprc": average_precision_score(labels, attack_probs),
            "ece": calculate_ece(all_probs, labels),
            "confusion_matrix": confusion_matrix(labels, all_preds).tolist()
        }

def find_models():
    import yaml
    # Base model
    models = {
        "Baseline": "outputs/mac_experiments/exp009/checkpoint/best"
    }
    
    # Find ablation models
    experiments = glob.glob("experiments/exp*")
    for exp in experiments:
        meta_file = os.path.join(exp, "metadata.json")
        exp_yaml = os.path.join(exp, "experiment.yaml")
        
        if os.path.exists(meta_file) and os.path.exists(exp_yaml):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            with open(exp_yaml, 'r') as f:
                exp_data = yaml.safe_load(f)
                
            desc = meta.get("description", "")
            seed = exp_data.get("data", {}).get("seed", 42)
            ckpt_path = os.path.join(exp, "checkpoint", "best")
            
            if not os.path.exists(ckpt_path): continue
            
            suffix = f"_seed{seed}" if seed != 42 else ""
            
            if "dataset_v2" in desc:
                models[f"Enron250{suffix}"] = ckpt_path
            elif "dataset_v3" in desc:
                models[f"Enron500{suffix}"] = ckpt_path
            elif "dataset_v4" in desc:
                models[f"Enron1000{suffix}"] = ckpt_path
    
    return models

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = find_models()
    if len(models) < 4:
        print(f"Warning: Did not find all 4 models. Found: {list(models.keys())}")
        
    # Load Datasets
    bipia_test = pd.read_csv("data/processed/dataset_v1/test.csv")
    handcrafted = pd.read_csv("data/processed/handcrafted_benign.csv")
    enron_heldout = pd.read_csv("data/processed/enron_heldout_test.csv")
    
    results = {}
    
    for name, path in models.items():
        print(f"\nEvaluating Model: {name} ({path})")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        model.eval()
        
        results[name] = {}
        
        print("  Running BIPIA Benchmark...")
        results[name]["BIPIA"] = evaluate(model, tokenizer, device, bipia_test, is_ood=False)
        
        print("  Running Handcrafted Benign OOD...")
        results[name]["Handcrafted"] = evaluate(model, tokenizer, device, handcrafted, is_ood=True)
        
        print("  Running Held-out Enron OOD...")
        results[name]["EnronHeldout"] = evaluate(model, tokenizer, device, enron_heldout, is_ood=True)
        
        # Cleanup memory
        del model
        del tokenizer
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
    os.makedirs("reports/enron_ablation", exist_ok=True)
    with open("reports/enron_ablation/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nEvaluations complete. Results saved to reports/enron_ablation/evaluation_results.json")

if __name__ == '__main__':
    main()
