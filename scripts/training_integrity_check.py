"""
Training Integrity Check — runs 10 batches end-to-end on GPU.
Uses GradScaler overflow detection (same as real training script).
"""
import os, sys, torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mm_taskgate import MMTaskGate
from training.losses import MultiTaskLoss
from data.mm_dataset import MMDataset


def main():
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available"); return
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ds = MMDataset("datasets/train.csv")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = MMTaskGate(num_tasks=3).to(device)
    cfg = {"lambda_fake_news": 1.0, "lambda_sentiment": 0.8,
           "lambda_harmful": 1.2, "focal_loss_gamma": 2.0}
    loss_fn = MultiTaskLoss(cfg).to(device)
    scaler  = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    neg = torch.tensor(-1, dtype=torch.long, device=device)

    ok_batches = 0
    nan_loss = 0
    overflow_skips = 0

    for i, batch in enumerate(loader):
        if i >= 10:
            break

        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        img  = batch["image"].to(device)
        hi   = batch["has_image"].to(device)
        tid  = batch["task_id"].to(device)
        lbl  = batch["label"].to(device)

        targets = {
            "labels_fake":      torch.where(tid == 0, lbl, neg),
            "labels_sentiment": torch.where(tid == 1, lbl, neg),
            "labels_harmful":   torch.where(tid == 2, lbl, neg),
        }

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(ids, mask, img, hi, tid)
            gates = {k: out[k] for k in ("token_gates", "task_gate", "modal_gate")}
            loss, _ = loss_fn(out, targets, gates, gate_lambda=0.01)

        if not torch.isfinite(loss):
            nan_loss += 1
            print(f"  batch {i}: NaN loss — skipped")
            continue

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [p for p in model.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scaler.get_scale() < old_scale:
            overflow_skips += 1
            print(f"  batch {i}: loss={loss.item():.4f}  AMP overflow (step skipped, normal)")
        else:
            ok_batches += 1
            gn_val = gn.item() if torch.isfinite(gn) else float('nan')
            print(f"  batch {i}: loss={loss.item():.4f}  grad_norm={gn_val:.4f}  OK")

    vram = torch.cuda.max_memory_reserved(0) / 1e9
    print(f"\nResults:")
    print(f"  Successful batches : {ok_batches}")
    print(f"  NaN loss batches   : {nan_loss}")
    print(f"  AMP overflow skips : {overflow_skips}")
    print(f"  GPU VRAM reserved  : {vram:.2f} GB")

    if ok_batches >= 5:
        print("\n✓ Integrity check PASSED — pipeline is stable.")
    elif ok_batches >= 2:
        print("\n⚠ Mostly OK — some AMP overflows are normal early in training.")
    else:
        print("\n✗ Too many failures — investigate loss/gradient sources.")


if __name__ == "__main__":
    main()
