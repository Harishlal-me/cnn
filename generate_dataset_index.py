import os
import json
import pandas as pd
import csv
from collections import Counter

base_dir = r"D:\multimodal-taskgate\datasets"
output_csv = os.path.join(base_dir, "dataset_index.csv")

records = []
status = {
    "Hateful Memes": {"status": "Missing", "loaded": 0},
    "Memotion 7K": {"status": "Missing", "loaded": 0},
    "MVSA-Single": {"status": "Missing", "loaded": 0},
    "HarM": {"status": "Missing", "loaded": 0},
    "HateXplain": {"status": "Missing", "loaded": 0},
}

# 1. Hateful Memes
hm_dir = os.path.join(base_dir, "data")
loaded_hm = 0
for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
    hm_path = os.path.join(hm_dir, split)
    if os.path.exists(hm_path):
        status["Hateful Memes"]["status"] = "Found"
        with open(hm_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                records.append({
                    "dataset_name": "hateful_memes",
                    "image_path": os.path.join("data", "img", os.path.basename(data.get("img", ""))),
                    "text": data.get("text", "").replace('\n', ' '),
                    "label": str(data.get("label", "")),
                    "task": "harmful"
                })
                loaded_hm += 1
status["Hateful Memes"]["loaded"] = loaded_hm

# 2. Memotion 7K
memo_path = os.path.join(base_dir, "memotion_dataset_7k", "labels.csv")
loaded_memo = 0
if os.path.exists(memo_path):
    status["Memotion 7K"]["status"] = "Found"
    df_memo = pd.read_csv(memo_path)
    for _, row in df_memo.iterrows():
        text = str(row.get('text_corrected', row.get('text_ocr', '')))
        records.append({
            "dataset_name": "memotion7k",
            "image_path": os.path.join("memotion_dataset_7k", "images", str(row.get('image_name', ''))).replace("\\", "/"),
            "text": text.replace('\n', ' '),
            "label": str(row.get('overall_sentiment', '')),
            "task": "sentiment"
        })
        loaded_memo += 1
status["Memotion 7K"]["loaded"] = loaded_memo

# 3. MVSA-Single
mvsa_label_path = os.path.join(base_dir, "MVSA_Single", "labelResultAll.txt")
mvsa_data_dir = os.path.join(base_dir, "MVSA_Single", "data")
loaded_mvsa = 0
if os.path.exists(mvsa_label_path):
    status["MVSA-Single"]["status"] = "Found"
    with open(mvsa_label_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                idx = parts[0]
                label = parts[1]
                txt_path = os.path.join(mvsa_data_dir, f"{idx}.txt")
                text = ""
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='latin-1', errors='ignore') as tf:
                        text = tf.read().strip()
                records.append({
                    "dataset_name": "mvsa_single",
                    "image_path": os.path.join("MVSA_Single", "data", f"{idx}.jpg").replace("\\", "/"),
                    "text": text.replace('\n', ' '),
                    "label": label,
                    "task": "sentiment"
                })
                loaded_mvsa += 1
status["MVSA-Single"]["loaded"] = loaded_mvsa

# 4. HarM (Harm-C and Harm-P)
loaded_harm = 0
for harm_variant in ["Harm-C", "Harm-P"]:
    harm_anno_dir = os.path.join(base_dir, "MINI_PROJECT_2", harm_variant, "datasets", "memes", "defaults", "annotations")
    harm_img_dir = os.path.join("MINI_PROJECT_2", harm_variant, "datasets", "memes", "defaults", "images")
    if os.path.exists(harm_anno_dir):
        status["HarM"]["status"] = "Found"
        for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            sp_path = os.path.join(harm_anno_dir, split)
            if os.path.exists(sp_path):
                with open(sp_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        labels = data.get("labels", [])
                        label_str = ",".join(labels) if isinstance(labels, list) else str(labels)
                        
                        img_name = data.get("image", data.get("img", ""))
                        records.append({
                            "dataset_name": "harm",
                            "image_path": os.path.join(harm_img_dir, img_name).replace("\\", "/"),
                            "text": str(data.get("text", "")).replace('\n', ' '),
                            "label": label_str,
                            "task": "harmful"
                        })
                        loaded_harm += 1
status["HarM"]["loaded"] = loaded_harm

# 5. HateXplain
hx_path = os.path.join(base_dir, "hatexplain", "dataset.json")
loaded_hx = 0
if os.path.exists(hx_path):
    status["HateXplain"]["status"] = "Found"
    with open(hx_path, 'r', encoding='utf-8') as f:
        hx_data = json.load(f)
        for post_id, data in hx_data.items():
            tokens = data.get("post_tokens", [])
            text = " ".join(tokens).replace('\n', ' ')
            
            # get majority label
            annotators = data.get("annotators", [])
            labels = [a.get("label") for a in annotators if "label" in a]
            majority_label = ""
            if labels:
                majority_label = Counter(labels).most_common(1)[0][0]
                
            records.append({
                "dataset_name": "hatexplain",
                "image_path": "", # text only
                "text": text,
                "label": majority_label,
                "task": "harmful"
            })
            loaded_hx += 1
status["HateXplain"]["loaded"] = loaded_hx

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Generated {output_csv} with {len(df)} records.\n")
print(f"{'Dataset':<20} | {'Status':<10} | {'Samples Loaded':<15}")
print("-" * 52)
for ds, info in status.items():
    print(f"{ds:<20} | {info['status']:<10} | {info['loaded']:<15}")
