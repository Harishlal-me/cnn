"""
MultiModal TaskGate Dataset — with OCR fallback, image augmentation,
and RoBERTa tokenizer.
"""
import ast, os, torch
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import RobertaTokenizer

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data.ocr_extractor import get_ocr_extractor

# ── Label maps ───────────────────────────────────────────────────────────────────
HARMFUL_NEG = {"not_offensive", "clean", "normal", "not hateful",
               "not harmful", "non-toxic", "benign", "0", "0.0"}
HARMFUL_POS = {"hatespeech", "offensive", "misogynous", "abusive",
               "hate_speech", "hateful", "toxic", "harmful",
               "very harmful", "1", "1.0"}
SENTIMENT_MAP = {"negative": 0, "very_negative": 0,
                 "neutral": 1,
                 "positive": 2, "very_positive": 2}


def _map_label(raw: str, task_id: int) -> int:
    raw = raw.strip().lower()
    if raw in ("", "nan"):
        return -1
    if "," in raw:
        raw = raw.split(",")[0].strip()

    if task_id == 2:  # harmful
        if raw in HARMFUL_NEG: return 0
        if raw in HARMFUL_POS: return 1
        try:
            v = int(float(raw)); return v if v in (0, 1) else -1
        except ValueError:
            return -1
    if task_id == 1:  # sentiment
        if raw in SENTIMENT_MAP: return SENTIMENT_MAP[raw]
        try:
            v = int(float(raw)); return v if v in (0, 1, 2) else -1
        except ValueError:
            return -1
    if task_id == 0:  # fake news
        if raw in ("fake", "1", "1.0"): return 1
        if raw in ("real", "0", "0.0"): return 0
        try:
            v = int(float(raw)); return v if v in (0, 1) else -1
        except ValueError:
            return -1
    return -1


# ── Transforms ───────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MMDataset(Dataset):
    def __init__(self, csv_file, max_length=128, is_train=False):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_length = max_length
        self.transform = TRAIN_TRANSFORM if is_train else EVAL_TRANSFORM
        self._task_map = {"fake_news": 0, "sentiment": 1, "harmful": 2}
        self._ocr = None  # lazy init

    def __len__(self):
        return len(self.data)

    def _fallback(self):
        return {
            "input_ids":      torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "image":          torch.zeros(3, 224, 224),
            "has_image":      torch.tensor(False, dtype=torch.bool),
            "task_id":        torch.tensor(-1, dtype=torch.long),
            "label":          torch.tensor(-1, dtype=torch.long),
        }

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]

            # ── Text ─────────────────────────────────────────────────────────
            text = str(row["text"]) if pd.notna(row["text"]) else ""
            if text.strip() == "" or text.lower() == "nan":
                text = ""

            # ── Image ────────────────────────────────────────────────────────
            img_path = str(row["image_path"]) if pd.notna(row["image_path"]) else ""
            has_image = False
            image_tensor = torch.zeros(3, 224, 224)
            if img_path and img_path.lower() != "nan" and os.path.isfile(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    image_tensor = self.transform(img)
                    has_image = True
                except Exception:
                    pass

            # ── OCR fallback for image-only samples ──────────────────────────
            if text == "" and has_image:
                try:
                    if self._ocr is None:
                        self._ocr = get_ocr_extractor()
                    text = self._ocr.extract(img_path)
                except Exception:
                    pass

            # ── Tokenize ─────────────────────────────────────────────────────
            encoded = self.tokenizer(
                text, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )

            # ── Task ─────────────────────────────────────────────────────────
            task_str = str(row["task"]).strip().lower()
            if task_str.startswith("["):
                try:
                    task_str = ast.literal_eval(task_str)[0]
                except Exception:
                    pass
            task_id = self._task_map.get(task_str, -1)

            # ── Label ────────────────────────────────────────────────────────
            raw_label = str(row["label"]) if pd.notna(row["label"]) else ""
            label = _map_label(raw_label, task_id)

            return {
                "input_ids":      encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "image":          image_tensor,
                "has_image":      torch.tensor(has_image, dtype=torch.bool),
                "task_id":        torch.tensor(task_id, dtype=torch.long),
                "label":          torch.tensor(label, dtype=torch.long),
            }
        except Exception:
            return self._fallback()
