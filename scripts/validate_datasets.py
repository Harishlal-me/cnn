import sys
import os
import torch
import pandas as pd
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mm_dataset import MMDataset

def validate_datasets():
    datasets_to_check = ["datasets/train.csv", "datasets/val.csv", "datasets/test.csv"]
    
    total_samples = 0
    invalid_rows = 0
    missing_images = 0
    label_dist = Counter()
    
    for split_path in datasets_to_check:
        if not os.path.exists(split_path):
            print(f"Skipping {split_path} (not found)")
            continue
            
        print(f"\nScanning: {split_path}...")
        dataset = MMDataset(split_path)
        
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                total_samples += 1
                
                # Check image presence
                if not item["has_image"].item():
                    missing_images += 1
                    
                # Collect label dist
                lbl = item["label"].item()
                label_dist[lbl] += 1
                
            except Exception as e:
                invalid_rows += 1
                print(f"Row {i} crashed in {split_path}: {e}")
                
    print("\n--- Validation Complete ---")
    print(f"Total samples: {total_samples}")
    print(f"Invalid/Crashed rows: {invalid_rows}")
    print(f"Missing images: {missing_images}")
    print(f"Label distribution: {dict(label_dist)}")

if __name__ == "__main__":
    validate_datasets()
