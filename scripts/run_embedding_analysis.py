import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from tqdm import tqdm
import os

def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the CLS token (index 0) hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embeddings)
            
    return torch.cat(embeddings, dim=0).numpy()

def main():
    model_path = "outputs/mac_experiments/exp009/checkpoint/best"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load BIPIA
    bipia_df = pd.read_csv("data/processed/dataset_v0/test.csv")
    
    # We just need a sample of 200 from each category to make a clean plot
    bipia_benign = bipia_df[bipia_df["label"] == 0].head(200).copy()
    bipia_attack = bipia_df[bipia_df["label"] == 1].head(200).copy()
    
    # Load Handcrafted
    hand_df = pd.read_csv("data/processed/handcrafted_benign.csv").head(200).copy()
    
    print("Extracting BIPIA Benign embeddings...")
    bipia_benign_texts = (bipia_benign["context"] if "context" in bipia_benign.columns else bipia_benign["text"]).tolist()
    emb_bipia_benign = get_embeddings(bipia_benign_texts, model, tokenizer, device)
    
    print("Extracting BIPIA Attack embeddings...")
    bipia_attack_texts = (bipia_attack["context"] if "context" in bipia_attack.columns else bipia_attack["text"]).tolist()
    emb_bipia_attack = get_embeddings(bipia_attack_texts, model, tokenizer, device)
    
    print("Extracting Handcrafted Benign embeddings...")
    emb_hand = get_embeddings(hand_df["text"].tolist(), model, tokenizer, device)
    
    # Combine and reduce
    import numpy as np
    all_embs = np.vstack([emb_bipia_benign, emb_bipia_attack, emb_hand])
    labels = ["BIPIA Benign"] * len(emb_bipia_benign) + ["BIPIA Attack"] * len(emb_bipia_attack) + ["Handcrafted Benign"] * len(emb_hand)
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_embs = reducer.fit_transform(all_embs)
    
    df_plot = pd.DataFrame({
        "UMAP 1": reduced_embs[:, 0],
        "UMAP 2": reduced_embs[:, 1],
        "Category": labels
    })
    
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_plot, x="UMAP 1", y="UMAP 2", hue="Category", 
                    palette={"BIPIA Benign": "green", "BIPIA Attack": "red", "Handcrafted Benign": "blue"}, 
                    alpha=0.7)
    plt.title("UMAP Projection of CLS Embeddings")
    plt.tight_layout()
    
    os.makedirs("paper_assets", exist_ok=True)
    plt.savefig("paper_assets/embedding_clusters_umap.png", dpi=300)
    print("Saved plot to paper_assets/embedding_clusters_umap.png")

if __name__ == "__main__":
    main()
