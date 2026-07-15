import pandas as pd
import os
import shutil
import re
import json
import uuid

def clean_email(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'-----Original Message-----.*', '', text, flags=re.DOTALL)
    text = re.sub(r'From:.*To:.*Subject:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_qa_relevant(text):
    # Filter for enterprise emails that support Question Answering
    keywords = [
        "meeting", "deadline", "update", "schedule", "project", 
        "attached", "report", "review", "agenda", "task", 
        "action", "status", "discuss", "approve", "forward",
        "budget", "manager", "team", "client"
    ]
    text_lower = text.lower()
    # Require at least 2 QA-relevant keywords to ensure it's a solid business email
    matches = sum(1 for k in keywords if k in text_lower)
    return matches >= 2

def main():
    print("Loading raw Enron data...")
    df_raw = pd.read_csv("data/raw/enron_spam.csv")
    
    df_benign = df_raw[df_raw['label_text'] == 'ham'].copy()
    df_benign['clean_text'] = df_benign['text'].apply(clean_email)
    
    # QA filter + length filter
    df_benign = df_benign[(df_benign['clean_text'].str.len() > 150) & (df_benign['clean_text'].str.len() < 3000)]
    df_benign = df_benign[df_benign['clean_text'].apply(is_qa_relevant)]
    
    df_benign = df_benign.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total QA-relevant Enron benign emails: {len(df_benign)}")
    if len(df_benign) < 2000:
        print("WARNING: Not enough emails after QA filter!")
        
    df_train_enron = df_benign.iloc[:1000].copy()
    df_test_enron = df_benign.iloc[1000:2000].copy()
    
    df_train_enron['id'] = ['enron_train_' + str(uuid.uuid4())[:8] for _ in range(len(df_train_enron))]
    df_test_enron['id'] = ['enron_test_' + str(uuid.uuid4())[:8] for _ in range(len(df_test_enron))]
    
    # Use `context` to match dataset_v1
    for df in [df_train_enron, df_test_enron]:
        df['label'] = 0
        df['source'] = 'enron_ablation'
        df['context'] = df['clean_text']
        
    df_test_enron[['id', 'context', 'label', 'source']].to_csv("data/processed/enron_heldout_test.csv", index=False)
    
    df_v1_train = pd.read_csv("data/processed/dataset_v1/train.csv")
    
    versions = {2: 250, 3: 500, 4: 1000}
    
    for v, count in versions.items():
        os.makedirs(f"data/processed/dataset_v{v}", exist_ok=True)
        df_enron_subset = df_train_enron.iloc[:count]
        
        # Concat
        df_new_train = pd.concat([df_v1_train, df_enron_subset[['id', 'context', 'label', 'source']]], ignore_index=True)
        # Ensure no NaNs in context
        df_new_train['context'] = df_new_train['context'].fillna("").astype(str)
        
        df_new_train.to_csv(f"data/processed/dataset_v{v}/train.csv", index=False)
        shutil.copy("data/processed/dataset_v1/validation.csv", f"data/processed/dataset_v{v}/validation.csv")
        shutil.copy("data/processed/dataset_v1/test.csv", f"data/processed/dataset_v{v}/test.csv")
        print(f"Created dataset_v{v} with {len(df_new_train)} rows")
        
    # Overwrite teacher_annotations.jsonl safely
    ann_path = "data/processed/teacher_annotations.jsonl"
    with open(ann_path, "r") as f:
        lines = f.readlines()
        
    cleaned_lines = [l for l in lines if '"teacher_reasoning": "[Ablation]' not in l]
    
    with open(ann_path, "w") as f:
        for l in cleaned_lines:
            f.write(l)
        for _, row in df_train_enron.iterrows():
            record = {
                "id": row['id'],
                "teacher_prediction": 0,
                # Avoid strict 0.0 to prevent log(0) NaN in KL Divergence Loss
                "teacher_probs": [0.9999, 0.0001],
                "teacher_entropy": 0.001,
                "teacher_reasoning": "[Ablation] Enron benign pseudo-label."
            }
            f.write(json.dumps(record) + "\n")
            
    print("Fixed dataset generation with QA constraint. Prevented NaN probabilities.")

if __name__ == '__main__':
    main()
