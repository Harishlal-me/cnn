"""
MultiModal TaskGate — Real Evaluation Pipeline
================================================
Loads the trained checkpoint, runs inference on the validation set,
and computes per-task F1 / Precision / Recall / Accuracy.
"""

import os, sys, torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mm_taskgate import MMTaskGate
from data.mm_dataset import MMDataset

# ─── Config ──────────────────────────────────────────────────────────────────────
CHECKPOINT   = "checkpoints/stage3_best.pt"
VAL_CSV      = "datasets/val.csv"
TEST_CSV     = "datasets/test.csv"
BATCH_SIZE   = 16
TASK_NAMES   = {0: "Fake News", 1: "Sentiment", 2: "Harmful Content"}
TASK_CLASSES  = {0: 2, 1: 3, 2: 2}   # num classes per task


def verify_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Evaluation cannot continue.")
    device = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU  : {p.name}")
    print(f"VRAM : {p.total_memory / 1e9:.2f} GB")
    return device


def load_model(device):
    model = MMTaskGate(num_tasks=3)

    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    # Handle different checkpoint key names
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError(f"Checkpoint keys: {list(ckpt.keys())} — cannot find model state dict")

    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")
    if "f1" in ckpt:
        print(f"  Training best F1 (placeholder): {ckpt['f1']:.4f}")
    if "epoch" in ckpt:
        print(f"  Saved at epoch: {ckpt['epoch']}")
    return model


def evaluate(model, csv_path, device):
    """Run inference and collect per-task predictions & labels."""
    print(f"\nEvaluating on: {csv_path}")

    dataset = MMDataset(csv_path)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"  Samples: {len(dataset)}  |  Batches: {len(loader)}")

    # Collect predictions per task
    all_preds  = defaultdict(list)   # task_id → list of predicted labels
    all_labels = defaultdict(list)   # task_id → list of true labels

    neg = torch.tensor(-1, dtype=torch.long, device=device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            img  = batch["image"].to(device, non_blocking=True)
            hi   = batch["has_image"].to(device, non_blocking=True)
            tid  = batch["task_id"].to(device, non_blocking=True)
            lbl  = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(ids, mask, img, hi, tid)

            # Map logits → task predictions
            logits_map = {
                0: out["logits_fake"],       # [B, 2]
                1: out["logits_sentiment"],  # [B, 3]
                2: out["logits_harmful"],    # [B, 2]
            }

            for task_id_val in (0, 1, 2):
                # Find samples in this batch that belong to this task
                task_mask = (tid == task_id_val)
                label_mask = (lbl >= 0) & task_mask   # valid labels only

                if label_mask.sum() == 0:
                    continue

                logits = logits_map[task_id_val][label_mask]   # [N, C]
                preds  = logits.argmax(dim=1).cpu().numpy()
                truths = lbl[label_mask].cpu().numpy()

                all_preds[task_id_val].extend(preds.tolist())
                all_labels[task_id_val].extend(truths.tolist())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(loader)} batches...")

    return all_preds, all_labels


def compute_and_print_metrics(all_preds, all_labels):
    """Compute per-task metrics and macro F1."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    task_f1s = []

    for task_id in sorted(TASK_NAMES.keys()):
        name = TASK_NAMES[task_id]
        preds  = np.array(all_preds.get(task_id, []))
        labels = np.array(all_labels.get(task_id, []))

        if len(preds) == 0:
            print(f"\n  {name}")
            print(f"    No samples found for this task.")
            continue

        n_classes = TASK_CLASSES[task_id]
        average  = "binary" if n_classes == 2 else "macro"

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average=average, zero_division=0)
        rec  = recall_score(labels, preds, average=average, zero_division=0)
        f1   = f1_score(labels, preds, average=average, zero_division=0)

        print(f"\n  {name}  ({len(preds)} samples)")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"    Confusion Matrix:")
        for row in cm:
            print(f"      {row}")

        # Detailed classification report
        print(f"\n    Classification Report:")
        report = classification_report(labels, preds, zero_division=0)
        for line in report.split("\n"):
            print(f"      {line}")

        task_f1s.append(f1)

    # Macro F1 across tasks
    macro_f1 = np.mean(task_f1s) if task_f1s else 0.0
    print(f"\n{'='*60}")
    print(f"  MACRO F1 (across tasks): {macro_f1:.4f}")
    print(f"{'='*60}")

    return macro_f1


def save_results(all_preds, all_labels, macro_f1):
    """Save evaluation results to a text file."""
    os.makedirs("results", exist_ok=True)
    path = "results/evaluation_results.txt"
    with open(path, "w") as f:
        f.write("MultiModal TaskGate — Evaluation Results\n")
        f.write("=" * 50 + "\n\n")

        for task_id in sorted(TASK_NAMES.keys()):
            name = TASK_NAMES[task_id]
            preds  = np.array(all_preds.get(task_id, []))
            labels = np.array(all_labels.get(task_id, []))

            if len(preds) == 0:
                f.write(f"{name}: No samples\n\n")
                continue

            n_classes = TASK_CLASSES[task_id]
            average  = "binary" if n_classes == 2 else "macro"

            acc  = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, average=average, zero_division=0)
            rec  = recall_score(labels, preds, average=average, zero_division=0)
            f1   = f1_score(labels, preds, average=average, zero_division=0)

            f.write(f"{name} ({len(preds)} samples)\n")
            f.write(f"  Accuracy  : {acc:.4f}\n")
            f.write(f"  Precision : {prec:.4f}\n")
            f.write(f"  Recall    : {rec:.4f}\n")
            f.write(f"  F1 Score  : {f1:.4f}\n\n")

        f.write(f"Macro F1: {macro_f1:.4f}\n")

    print(f"\nResults saved to: {path}")


def main():
    device = verify_gpu()
    model  = load_model(device)

    # Evaluate on validation set
    print("\n" + "-" * 60)
    print("  VALIDATION SET")
    print("-" * 60)
    val_preds, val_labels = evaluate(model, VAL_CSV, device)
    macro_f1 = compute_and_print_metrics(val_preds, val_labels)

    # Evaluate on test set if available
    if os.path.isfile(TEST_CSV):
        print("\n" + "-" * 60)
        print("  TEST SET")
        print("-" * 60)
        test_preds, test_labels = evaluate(model, TEST_CSV, device)
        compute_and_print_metrics(test_preds, test_labels)

    # Save results
    save_results(val_preds, val_labels, macro_f1)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
