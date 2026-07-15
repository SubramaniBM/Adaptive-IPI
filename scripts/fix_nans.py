import pandas as pd
import glob

def main():
    files = glob.glob("data/processed/dataset_v*/train.csv")
    for f in files:
        df = pd.read_csv(f)
        if df['text'].isnull().any():
            print(f"Fixing NaNs in {f}")
            df['text'] = df['text'].fillna("")
            # Ensure everything is a string
            df['text'] = df['text'].astype(str)
            df.to_csv(f, index=False)
    
    # Also check heldout
    f = "data/processed/enron_heldout_test.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        if df['text'].isnull().any():
            df['text'] = df['text'].fillna("")
            df['text'] = df['text'].astype(str)
            df.to_csv(f, index=False)
            
    print("Fixed all NaN values.")

if __name__ == '__main__':
    import os
    main()
