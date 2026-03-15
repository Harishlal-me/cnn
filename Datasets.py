"""
MultiModal TaskGate — Dataset Loaders
Supports: Hateful Memes, Memotion 7K, MVSA-Single, HarM, HateXplain
"""

import os
import json
import jsonlines
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from transformers import DistilBertTokenizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_SEQ_LEN = 128
IMAGE_SIZE = 224

TASK_LABELS = {
    "fake_news": 0,
    "sentiment": 1,
    "harmful":   2,
}

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────

def get_transform(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ─────────────────────────────────────────────
# BASE DATASET CLASS
# ─────────────────────────────────────────────

class BaseMultiModalDataset(Dataset):
    """
    Base class for all datasets.
    Each sample returns:
      input_ids      : [MAX_SEQ_LEN]       tokenized text
      attention_mask : [MAX_SEQ_LEN]       attention mask
      image          : [3 x 224 x 224]     image tensor (or zeros if no image)
      has_image      : bool                True if image exists
      label_fake     : int or -1           fake news label (-1 = not available)
      label_sentiment: int or -1           sentiment label (-1 = not available)
      label_harmful  : int or -1           harmful label  (-1 = not available)
    """

    def tokenize(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "[UNK]"
        encoded = TOKENIZER(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
        )

    def load_image(self, image_path, transform):
        import torch
        try:
            img = Image.open(image_path).convert("RGB")
            return transform(img), True
        except Exception:
            # Return zero tensor if image missing or corrupt
            import torch
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), False



# ─────────────────────────────────────────────
# 2. HATEFUL MEMES (Facebook)
# ─────────────────────────────────────────────
# Folder structure:
#   data/
#     img/             ← images named e.g. 12345.png
#     train.jsonl      ← {"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}
#     dev.jsonl
#     test.jsonl
#
# label: 0 = not hateful, 1 = hateful

class HatefulMemesDataset(BaseMultiModalDataset):

    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.transform = get_transform(split)

        split_file_map = {
            "train": "train.jsonl",
            "val":   "dev.jsonl",
            "test":  "test.jsonl",
        }
        jsonl_path = os.path.join(root_dir, split_file_map[split])

        self.samples = []
        with jsonlines.open(jsonl_path) as reader:
            for obj in reader:
                self.samples.append({
                    "img_path": os.path.join(root_dir, obj["img"]),
                    "text":     obj["text"],
                    "harmful":  obj["label"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]
        input_ids, attention_mask = self.tokenize(sample["text"])
        image, has_image = self.load_image(sample["img_path"], self.transform)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "image":           image,
            "has_image":       torch.tensor(has_image, dtype=torch.bool),
            "label_fake":      torch.tensor(-1,                dtype=torch.long),
            "label_sentiment": torch.tensor(-1,                dtype=torch.long),
            "label_harmful":   torch.tensor(sample["harmful"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 3. MEMOTION 7K
# ─────────────────────────────────────────────
# Folder structure:
#   memotion_dataset_7k/
#     images/          ← image files
#     labels.csv       ← image_name, overall_sentiment, humour, sarcasm, offensive, ...
#
# Sentiment mapping:
#   "positive" → 0, "negative" → 1, "neutral" → 2
#
# Harmful mapping (offensive column):
#   "not_offensive"    → 0
#   "slight"           → 1
#   "very_offensive"   → 1
#   "hateful_offensive"→ 1

class Memotion7KDataset(BaseMultiModalDataset):

    SENTIMENT_MAP = {"positive": 0, "negative": 1, "neutral": 2}
    OFFENSIVE_MAP = {
        "not_offensive":     0,
        "slight":            1,
        "very_offensive":    1,
        "hateful_offensive": 1,
    }

    def __init__(self, root_dir, split="train", split_ratio=(0.7, 0.1, 0.2)):
        self.root_dir = root_dir
        self.img_dir  = os.path.join(root_dir, "images")
        self.transform = get_transform(split)

        df = pd.read_csv(os.path.join(root_dir, "labels.csv"))

        self.samples = []
        for _, row in df.iterrows():
            sentiment_raw = str(row.get("overall_sentiment", "")).strip().lower()
            offensive_raw = str(row.get("offensive", "")).strip().lower()

            sentiment = self.SENTIMENT_MAP.get(sentiment_raw, -1)
            harmful   = self.OFFENSIVE_MAP.get(offensive_raw, -1)

            # Skip if both labels are unknown
            if sentiment == -1 and harmful == -1:
                continue

            self.samples.append({
                "img_name":  str(row["image_name"]).strip(),
                "text":      str(row.get("text_ocr", row.get("text_corrected", ""))).strip(),
                "sentiment": sentiment,
                "harmful":   harmful,
            })

        # Split
        n = len(self.samples)
        train_end = int(n * split_ratio[0])
        val_end   = int(n * (split_ratio[0] + split_ratio[1]))

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]
        input_ids, attention_mask = self.tokenize(sample["text"])

        img_path = os.path.join(self.img_dir, sample["img_name"])
        image, has_image = self.load_image(img_path, self.transform)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "image":           image,
            "has_image":       torch.tensor(has_image, dtype=torch.bool),
            "label_fake":      torch.tensor(-1,                  dtype=torch.long),
            "label_sentiment": torch.tensor(sample["sentiment"], dtype=torch.long),
            "label_harmful":   torch.tensor(sample["harmful"],   dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 4. MVSA-SINGLE
# ─────────────────────────────────────────────
# Folder structure:
#   MVSA_Single/
#     data/              ← images named by ID (e.g. 1.jpg)
#     labelResultAll.txt ← lines: "ID\tlabel"
#                          label: positive / negative / neutral
#
# Sentiment mapping:
#   positive → 0, negative → 1, neutral → 2

class MVSASingleDataset(BaseMultiModalDataset):

    SENTIMENT_MAP = {"positive": 0, "negative": 1, "neutral": 2}

    def __init__(self, root_dir, split="train", split_ratio=(0.7, 0.1, 0.2)):
        self.root_dir  = root_dir
        self.img_dir   = os.path.join(root_dir, "data")
        self.transform = get_transform(split)

        label_path = os.path.join(root_dir, "labelResultAll.txt")
        self.samples = []

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("ID"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                sample_id  = parts[0].strip()
                label_raw  = parts[1].strip().lower()
                sentiment  = self.SENTIMENT_MAP.get(label_raw, -1)
                if sentiment == -1:
                    continue
                self.samples.append({
                    "id":        sample_id,
                    "sentiment": sentiment,
                })

        # Split
        n = len(self.samples)
        train_end = int(n * split_ratio[0])
        val_end   = int(n * (split_ratio[0] + split_ratio[1]))

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]

        # MVSA images are named by ID, try common extensions
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(self.img_dir, f"{sample['id']}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path:
            image, has_image = self.load_image(img_path, self.transform)
        else:
            import torch
            image, has_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), False

        # MVSA does not include text directly — use empty string
        # Text comes from the image via OCR at inference time
        input_ids, attention_mask = self.tokenize("")

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "image":           image,
            "has_image":       torch.tensor(has_image, dtype=torch.bool),
            "label_fake":      torch.tensor(-1,                  dtype=torch.long),
            "label_sentiment": torch.tensor(sample["sentiment"], dtype=torch.long),
            "label_harmful":   torch.tensor(-1,                  dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 5. HarM
# ─────────────────────────────────────────────
# Folder structure:
#   MINI_PROJECT_2/
#     Harm-C/            ← COVID harmful memes
#       labels.json or annotations.json
#       images/
#     Harm-P/            ← Political harmful memes
#       labels.json or annotations.json
#       images/
#
# label: harmful → 1, not_harmful → 0

class HarMDataset(BaseMultiModalDataset):

    def __init__(self, root_dir, split="train", split_ratio=(0.7, 0.1, 0.2)):
        self.transform = get_transform(split)
        self.samples = []

        split_file = f"{split}.jsonl"

        for subset in ["Harm-C", "Harm-P"]:
            subset_dir = os.path.join(root_dir, subset)
            if not os.path.exists(subset_dir):
                continue

            anno_dir = os.path.join(subset_dir, "datasets", "memes", "defaults", "annotations")
            img_dir = os.path.join(subset_dir, "datasets", "memes", "defaults", "images")
            label_file = os.path.join(anno_dir, split_file)

            if not os.path.exists(label_file):
                print(f"[HarM] Warning: label memory {label_file} not found")
                continue

            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    img_name = item.get("image", "")
                    text     = str(item.get("text", "")).replace("\n", " ")
                    labels   = item.get("labels", [])
                    label_str = ",".join(labels) if isinstance(labels, list) else str(labels)
                    
                    # Normalize label to 0/1 based on presence of "harmful" / "hateful"
                    is_harmful = 1 if "harmful" in label_str.lower() or "hateful" in label_str.lower() else 0

                    self.samples.append({
                        "img_path": os.path.join(img_dir, img_name),
                        "text":     text,
                        "harmful":  int(is_harmful),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]
        input_ids, attention_mask = self.tokenize(sample["text"])
        image, has_image = self.load_image(sample["img_path"], self.transform)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "image":           image,
            "has_image":       torch.tensor(has_image, dtype=torch.bool),
            "label_fake":      torch.tensor(-1,                dtype=torch.long),
            "label_sentiment": torch.tensor(-1,                dtype=torch.long),
            "label_harmful":   torch.tensor(sample["harmful"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 6. HateXplain (Text Only)
# ─────────────────────────────────────────────
# Folder structure:
#   hatexplain/
#     dataset.json   ← {"post_id": {"post_tokens": [...], "annotators": [{"label": "..."}]}}
#
# Label mapping (majority vote):
#   hatespeech → 1, offensive → 1, normal → 0

class HateXplainDataset(BaseMultiModalDataset):

    LABEL_MAP = {"hatespeech": 1, "offensive": 1, "normal": 0}

    def __init__(self, root_dir, split="train", split_ratio=(0.7, 0.1, 0.2)):
        import torch
        self.transform = get_transform(split)
        self.samples = []

        dataset_path = os.path.join(root_dir, "dataset.json")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        for post_id, item in data.items():
            tokens = item.get("post_tokens", [])
            text   = " ".join(tokens)

            annotators = item.get("annotators", [])
            labels = [
                self.LABEL_MAP.get(a.get("label", "normal").lower(), 0)
                for a in annotators
            ]
            if not labels:
                continue
            # Majority vote
            harmful = 1 if sum(labels) > len(labels) / 2 else 0

            self.samples.append({
                "text":    text,
                "harmful": harmful,
            })

        # Split
        n = len(self.samples)
        train_end = int(n * split_ratio[0])
        val_end   = int(n * (split_ratio[0] + split_ratio[1]))

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        sample = self.samples[idx]
        input_ids, attention_mask = self.tokenize(sample["text"])

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "image":           torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
            "has_image":       torch.tensor(False, dtype=torch.bool),
            "label_fake":      torch.tensor(-1,                dtype=torch.long),
            "label_sentiment": torch.tensor(-1,                dtype=torch.long),
            "label_harmful":   torch.tensor(sample["harmful"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# COMBINED DATASET BUILDER
# ─────────────────────────────────────────────

def build_dataset(data_root, split="train"):
    """
    Builds and returns a ConcatDataset from all available datasets.

    Expected folder structure:
      data_root/
        data/               ← Hateful Memes
        memotion_dataset_7k/
        MVSA_Single/
        MINI_PROJECT_2/     ← HarM
        hatexplain/         ← HateXplain
    """
    datasets = []

    path_map = {
        "HatefulMemes":        (HatefulMemesDataset,"data"),
        "Memotion7K":          (Memotion7KDataset,  "memotion_dataset_7k"),
        "MVSASingle":          (MVSASingleDataset,  "MVSA_Single"),
        "HarM":                (HarMDataset,        "MINI_PROJECT_2"),
        "HateXplain":          (HateXplainDataset,  "hatexplain"),
    }

    for name, (cls, folder) in path_map.items():
        full_path = os.path.join(data_root, folder)
        if os.path.exists(full_path):
            try:
                ds = cls(full_path, split=split)
                datasets.append(ds)
                print(f"[Dataset] {name}: {len(ds)} samples ({split})")
            except Exception as e:
                print(f"[Dataset] {name}: FAILED — {e}")
        else:
            print(f"[Dataset] {name}: path not found — {full_path}")

    if not datasets:
        raise RuntimeError("No datasets loaded. Check data_root path.")

    combined = ConcatDataset(datasets)
    print(f"\n[Dataset] Total {split} samples: {len(combined)}\n")
    return combined


# ─────────────────────────────────────────────
# DATALOADER BUILDER
# ─────────────────────────────────────────────

def build_dataloader(data_root, split="train", batch_size=32, num_workers=2):
    dataset = build_dataset(data_root, split=split)
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    DATA_ROOT = "./datasets"   # change this to your actual path

    print("=" * 50)
    print("Testing data loaders...")
    print("=" * 50)

    for split in ["train", "val", "test"]:
        loader = build_dataloader(DATA_ROOT, split=split, batch_size=4)
        batch  = next(iter(loader))

        print(f"\n── {split.upper()} BATCH ──")
        print(f"  input_ids:       {batch['input_ids'].shape}")
        print(f"  attention_mask:  {batch['attention_mask'].shape}")
        print(f"  image:           {batch['image'].shape}")
        print(f"  has_image:       {batch['has_image']}")
        print(f"  label_fake:      {batch['label_fake']}")
        print(f"  label_sentiment: {batch['label_sentiment']}")
        print(f"  label_harmful:   {batch['label_harmful']}")

    print("\n✅ All data loaders working correctly.")