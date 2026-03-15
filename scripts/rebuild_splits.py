"""Rebuild stratified 80/10/10 splits after fake news integration."""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS   = os.path.join(ROOT, "datasets")

df = pd.read_csv(os.path.join(DS, "dataset_index.csv"))
print(f"Total rows: {len(df)}")
print(f"Task distribution:\n{df['task'].value_counts().to_string()}\n")

# Create stratification key  (task × label)
df["_strat"] = df["task"].astype(str) + "_" + df["label"].astype(str)

# Drop groups with < 2 members (can't stratify)
counts = df["_strat"].value_counts()
rare = counts[counts < 2].index
if len(rare):
    print(f"Dropping {len(df[df['_strat'].isin(rare)])} rows from {len(rare)} rare strat groups")
    df = df[~df["_strat"].isin(rare)]

train_df, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_strat"])
val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["_strat"])

train_df = train_df.drop(columns=["_strat"])
val_df   = val_df.drop(columns=["_strat"])
test_df  = test_df.drop(columns=["_strat"])

train_df.to_csv(os.path.join(DS, "train.csv"), index=False)
val_df.to_csv(os.path.join(DS, "val.csv"), index=False)
test_df.to_csv(os.path.join(DS, "test.csv"), index=False)

print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

for t in sorted(df["task"].dropna().unique()):
    n = len(train_df[train_df["task"] == t])
    print(f"  {t}: {n} train samples")
