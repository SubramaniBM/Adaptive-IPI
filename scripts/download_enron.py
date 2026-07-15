import os
import pandas as pd
from datasets import load_dataset

def main():
    print("Downloading Enron dataset from Hugging Face...")
    ds = load_dataset('SetFit/enron_spam')
    
    os.makedirs('data/raw', exist_ok=True)
    
    # We will combine train and test to give the user the full corpus
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas()
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Keep only the relevant columns for standard NLP tasks
    df = df[['message_id', 'subject', 'message', 'text', 'label_text', 'date']]
    
    output_path = 'data/raw/enron_spam.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Successfully downloaded and saved {len(df)} emails to {output_path}")
    print("\nDataset Sample:")
    print(df[['subject', 'label_text']].head(5))

if __name__ == '__main__':
    main()
