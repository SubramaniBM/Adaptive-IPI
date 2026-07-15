import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def construct_input_and_baseline(text, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    ref_input_ids = torch.zeros_like(input_ids).to(device) # Baseline is padding/zero token
    ref_input_ids[0, 0] = tokenizer.cls_token_id
    ref_input_ids[0, -1] = tokenizer.sep_token_id
    
    attention_mask = inputs["attention_mask"].to(device)
    return input_ids, ref_input_ids, attention_mask

def explain_prediction(model, tokenizer, text, device, target_class=1):
    input_ids, ref_input_ids, attention_mask = construct_input_and_baseline(text, tokenizer, device)
    
    def forward_func(inputs, attention_mask=None):
        return model(inputs, attention_mask=attention_mask).logits
    
    # Use tok_embeddings for attribution in ModernBERT
    lig = LayerIntegratedGradients(forward_func, model.model.embeddings.tok_embeddings)
    
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(attention_mask,),
                                        target=target_class,
                                        return_convergence_delta=True)
    
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions) # Normalize
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Store token and attribution (ignore special tokens)
    results = []
    for token, attr in zip(tokens, attributions.cpu().numpy()):
        if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            # Clean ModernBERT token (remove leading 'Ġ' or similar if necessary, though ModernBERT might use standard BPE)
            clean_token = token.replace('Ġ', '')
            results.append((clean_token, float(attr)))
            
    return results

def plot_attributions(attributions, title, save_path):
    # Sort by absolute attribution to show the most important tokens
    sorted_attrs = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)[:15]
    tokens, scores = zip(*sorted_attrs)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if s > 0 else 'blue' for s in scores]
    sns.barplot(x=list(scores), y=list(tokens), palette=colors)
    plt.title(title)
    plt.xlabel('Attribution Score (Red: Push to Attack, Blue: Push to Benign)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    model_path = "outputs/mac_experiments/exp009/checkpoint/best"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    os.makedirs("reports/distribution_analysis", exist_ok=True)
    os.makedirs("paper_assets", exist_ok=True)
    
    # Load Handcrafted False Positives
    fp_df = pd.read_csv("reports/distribution_analysis/handcrafted_false_positives.csv")
    if len(fp_df) > 0:
        sample_fp = fp_df.iloc[0]["text"]
        print("Explaining Handcrafted False Positive...")
        fp_attrs = explain_prediction(model, tokenizer, sample_fp, device, target_class=1)
        plot_attributions(fp_attrs, "Integrated Gradients: Handcrafted False Positive", "paper_assets/ig_false_positive.png")
    
    # Load BIPIA True Negatives
    bipia_df = pd.read_csv("data/processed/dataset_v0/test.csv")
    tn_df = bipia_df[bipia_df["label"] == 0]
    if len(tn_df) > 0:
        sample_tn = tn_df.iloc[0]["context"] if "context" in tn_df.columns else tn_df.iloc[0]["text"]
        print("Explaining BIPIA True Negative...")
        tn_attrs = explain_prediction(model, tokenizer, sample_tn, device, target_class=1)
        plot_attributions(tn_attrs, "Integrated Gradients: BIPIA True Negative", "paper_assets/ig_true_negative.png")
        
    print("Explainability analysis complete. Saved to paper_assets/")

if __name__ == "__main__":
    main()
