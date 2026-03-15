import json
import pandas as pd
import os

base_dir = r"D:\multimodal-taskgate\datasets"
out_file = os.path.join(base_dir, "explore_output.txt")

with open(out_file, 'w', encoding='utf-8') as out:
    # Hateful Memes
    with open(os.path.join(base_dir, r"data\train.jsonl"), 'r') as f:
        out.write("Hateful Memes: " + f.readline().strip() + "\n")

    # Memotion 7K
    df = pd.read_csv(os.path.join(base_dir, r"memotion_dataset_7k\labels.csv"))
    out.write("\nMemotion columns: " + str(df.columns.tolist()) + "\n")
    out.write("Memotion sample: " + str(df.iloc[0].to_dict()) + "\n")

    # MVSA-Single
    with open(os.path.join(base_dir, r"MVSA_Single\labelResultAll.txt"), 'r') as f:
        out.write("\nMVSA label head: " + str([next(f).strip() for _ in range(3)]) + "\n")

    with open(os.path.join(base_dir, r"MVSA_Single\data\1.txt"), 'r', encoding='latin-1', errors='ignore') as f:
        out.write("MVSA text sample: " + f.read().strip() + "\n")

    # HarM
    harm_path = os.path.join(base_dir, r"MINI_PROJECT_2\Harm-C\datasets\memes\defaults\annotations\train.jsonl")
    if os.path.exists(harm_path):
        with open(harm_path, 'r', encoding='utf-8') as f:
            out.write("\nHarM sample: " + f.readline().strip() + "\n")
    else:
        out.write("\nHarM train.jsonl not found.\n")

    # HateXplain
    hx_path = os.path.join(base_dir, r"hatexplain\dataset.json")
    if os.path.exists(hx_path):
        with open(hx_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            first_key = list(data.keys())[0]
            out.write("\nHateXplain sample: " + str(data[first_key]) + "\n")
