import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_splits(index_path="datasets/dataset_index.csv", out_dir="datasets"):
    if not os.path.exists(index_path):
        print(f"ERROR: {index_path} not found.")
        return
        
    df = pd.read_csv(index_path)
    os.makedirs(out_dir, exist_ok=True)
    
    has_labels = "label" in df.columns and "task" in df.columns
    if has_labels:
        df["stratify_col"] = df["task"].astype(str) + "_" + df["label"].astype(str)
        counts = df["stratify_col"].value_counts()
        valid_stratify = df["stratify_col"].isin(counts[counts > 2].index)
        stratify_series = df.loc[valid_stratify, "stratify_col"]
    else:
        valid_stratify = pd.Series([False]*len(df))
        
    try:
        if valid_stratify.sum() == len(df):
            train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["stratify_col"])
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["stratify_col"])
        else:
            train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    except Exception as e:
        print("Stratification failed, falling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
        
    if "stratify_col" in train_df.columns:
        train_df = train_df.drop(columns=["stratify_col"])
        val_df = val_df.drop(columns=["stratify_col"])
        test_df = test_df.drop(columns=["stratify_col"])
        
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    
    print("Total samples:", len(df))
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

if __name__ == "__main__":
    prepare_splits()
