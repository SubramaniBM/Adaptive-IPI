import pandas as pd
import json
import re
import os

def analyze_dataset(df):
    text_col = "text" if "text" in df.columns else "context"
    stats = {}
    
    # Text length stats
    lengths = df[text_col].str.len()
    stats["avg_length"] = lengths.mean()
    
    # Sentence count approx (counting periods/newlines)
    sentence_counts = df[text_col].apply(lambda x: len(re.split(r'[.!?\n]+', str(x))) - 1)
    stats["avg_sentences"] = sentence_counts.mean()
    
    # Keywords
    keywords = [
        "summarize", "review", "search", "send", "forward", 
        "ignore", "system", "assistant", "prompt", "schedule", "meeting"
    ]
    
    for kw in keywords:
        stats[f"pct_{kw}"] = df[text_col].str.contains(r'\b' + kw + r'\b', case=False, regex=True).mean() * 100
        
    # Imperative verbs (simple heuristic: starts with verb or contains "please [verb]")
    imperatives = ["please", "kindly", "action required", "urgent"]
    stats["pct_imperative_cues"] = df[text_col].str.contains('|'.join(imperatives), case=False, regex=True).mean() * 100
    
    # AI/Security related
    ai_words = ["ai", "bot", "assistant", "prompt", "llm", "model", "gpt"]
    sec_words = ["security", "password", "confidential", "secret", "hack", "bypass"]
    
    stats["pct_ai_words"] = df[text_col].str.contains(r'\b(?:' + '|'.join(ai_words) + r')\b', case=False, regex=True).mean() * 100
    stats["pct_security_words"] = df[text_col].str.contains(r'\b(?:' + '|'.join(sec_words) + r')\b', case=False, regex=True).mean() * 100
    
    return stats

def main():
    print("Loading datasets...")
    # Load BIPIA Test Set
    bipia_df = pd.read_csv("data/processed/dataset_v0/test.csv")
    bipia_benign = bipia_df[bipia_df["label"] == 0].copy()
    bipia_attack = bipia_df[bipia_df["label"] == 1].copy()
    
    # Load Handcrafted Benign
    hand_df = pd.read_csv("data/processed/handcrafted_benign.csv")
    
    print("Analyzing BIPIA Benign...")
    bipia_benign_stats = analyze_dataset(bipia_benign)
    
    print("Analyzing BIPIA Attack...")
    bipia_attack_stats = analyze_dataset(bipia_attack)
    
    print("Analyzing Handcrafted Benign...")
    hand_stats = analyze_dataset(hand_df)
    
    # Save results
    results = {
        "bipia_benign": bipia_benign_stats,
        "bipia_attack": bipia_attack_stats,
        "handcrafted_benign": hand_stats
    }
    
    os.makedirs("reports/distribution_analysis", exist_ok=True)
    with open("reports/distribution_analysis/stats.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Print comparison
    print("\n--- DISTRIBUTION COMPARISON ---")
    print(f"{'Feature':<25} | {'BIPIA Benign':<15} | {'BIPIA Attack':<15} | {'Handcrafted Benign':<15}")
    print("-" * 75)
    for k in bipia_benign_stats.keys():
        print(f"{k:<25} | {bipia_benign_stats[k]:<15.2f} | {bipia_attack_stats[k]:<15.2f} | {hand_stats[k]:<15.2f}")

if __name__ == "__main__":
    main()
