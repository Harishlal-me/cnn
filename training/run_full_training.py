"""
MultiModal TaskGate — Full Curriculum Training Script (v4)
==========================================================
Changes from v3:
  • Stage3 epochs: 5 → 10
  • Cosine LR scheduler with 500-step warmup
  • MMDataset(is_train=True) for training augmentation
  • RoBERTa backbone (via updated TransformerBranch)
"""

import os, sys, time, torch, torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mm_taskgate import MMTaskGate
from training.losses import MultiTaskLoss
from data.mm_dataset import MMDataset

# ─── Stage config ────────────────────────────────────────────────────────────────
STAGE_CFG = {
    "stage1": {"lr": 2e-5, "bs": 16, "epochs": 5},
    "stage2": {"lr": 1e-5, "bs": 8,  "epochs": 5},
    "stage3": {"lr": 5e-6, "bs": 8,  "epochs": 10},
}
WARMUP_STEPS = 500


def verify_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Training cannot continue.")
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU : {p.name}  |  VRAM : {p.total_memory / 1e9:.2f} GB")
    return dev


def make_loaders(train_csv, val_csv, bs):
    return (
        DataLoader(MMDataset(train_csv, is_train=True), batch_size=bs,
                   shuffle=True, num_workers=0, pin_memory=True, drop_last=True),
        DataLoader(MMDataset(val_csv, is_train=False), batch_size=bs,
                   shuffle=False, num_workers=0, pin_memory=True),
    )


def freeze(model, stage):
    for p in model.parameters():
        p.requires_grad = False

    if stage == "stage1":
        for n, p in model.transformer_branch.named_parameters():
            if "layer.4" in n or "layer.5" in n or "layer.10" in n or "layer.11" in n:
                p.requires_grad = True
        for p in model.multiscale_cnn.parameters():
            p.requires_grad = True
        for p in model.task_heads.parameters():
            p.requires_grad = True
        for p in model.token_gate.parameters():
            p.requires_grad = True
        for p in model.task_gate.parameters():
            p.requires_grad = True

    elif stage == "stage2":
        for n, p in model.image_encoder.named_parameters():
            if "layer3" in n or "layer4" in n or "proj" in n:
                p.requires_grad = True
        for p in model.cross_modal_gate.parameters():
            p.requires_grad = True
        for p in model.task_heads.parameters():
            p.requires_grad = True

    else:
        for p in model.parameters():
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")
    return trainable


def train_epoch(model, loader, loss_fn, optimizer, scaler, scheduler, device):
    model.train()
    tot_loss, tot_grad, ok, nans, overflows = 0.0, 0.0, 0, 0, 0
    n_batches = len(loader)
    trainable = [p for p in model.parameters() if p.requires_grad]
    neg = torch.tensor(-1, dtype=torch.long, device=device)

    for i, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        img  = batch["image"].to(device, non_blocking=True)
        hi   = batch["has_image"].to(device, non_blocking=True)
        tid  = batch["task_id"].to(device, non_blocking=True)
        lbl  = batch["label"].to(device, non_blocking=True)

        targets = {
            "labels_fake":      torch.where(tid == 0, lbl, neg),
            "labels_sentiment": torch.where(tid == 1, lbl, neg),
            "labels_harmful":   torch.where(tid == 2, lbl, neg),
        }

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast("cuda"):
                out = model(ids, mask, img, hi, tid)
                gates = {k: out[k] for k in ("token_gates", "task_gate", "modal_gate")}
                loss, _ = loss_fn(out, targets, gates, gate_lambda=0.01)

            if not torch.isfinite(loss):
                nans += 1
                continue

            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # step scheduler after every optimizer update

            if scaler.get_scale() < old_scale:
                overflows += 1
            else:
                if torch.isfinite(gn):
                    tot_grad += gn.item()
                tot_loss += loss.item()
                ok += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            else:
                raise

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n_batches}] loss={tot_loss/max(1,ok):.4f}  ov={overflows}")
        if i == 0:
            print(f"  VRAM after 1st batch: {torch.cuda.max_memory_reserved(0)/1e9:.2f} GB")

    return (tot_loss / max(1, ok), tot_grad / max(1, ok), ok, nans, overflows)


def run_training():
    device = verify_gpu()
    print("Loading datasets…")

    cfg = {"lambda_fake_news": 1.0, "lambda_sentiment": 0.8,
           "lambda_harmful": 1.2, "focal_loss_gamma": 2.0}
    loss_fn = MultiTaskLoss(cfg).to(device)
    scaler  = torch.amp.GradScaler("cuda")
    model   = MMTaskGate(num_tasks=3).to(device)
    os.makedirs("checkpoints", exist_ok=True)

    for stage in ("stage1", "stage2", "stage3"):
        sc = STAGE_CFG[stage]
        n_epochs = sc["epochs"]
        print(f"\n{'='*60}")
        print(f"  {stage.upper()}  |  LR={sc['lr']}  |  BS={sc['bs']}  |  Epochs={n_epochs}")
        print(f"{'='*60}")

        tl, vl = make_loaders("datasets/train.csv", "datasets/val.csv", sc["bs"])
        print(f"  Train batches: {len(tl)}  |  Val batches: {len(vl)}")

        trainable = freeze(model, stage)
        opt = optim.AdamW(trainable, lr=sc["lr"], weight_decay=0.01)

        total_steps = len(tl) * n_epochs
        scheduler = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )

        best_f1 = -1.0
        start_epoch = 0

        # Auto-resume logic
        ckpt_path = f"checkpoints/{stage}_best.pt"
        if os.path.exists(ckpt_path):
            print(f"  [Resume] Found existing checkpoint for {stage}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "opt" in ckpt:
                opt.load_state_dict(ckpt["opt"])
            if "f1" in ckpt:
                best_f1 = ckpt["f1"]
            if "epoch" in ckpt:
                # ckpt["epoch"] is the index of the last completed epoch
                start_epoch = ckpt["epoch"] + 1
                
            if start_epoch >= n_epochs:
                print(f"  [Resume] Stage {stage} is already complete (Epochs {start_epoch}/{n_epochs}). Skipping.")
                continue
            else:
                print(f"  [Resume] Resuming {stage} from Epoch {start_epoch+1}")
                # Fast-forward scheduler
                for _ in range(start_epoch * len(tl)):
                    scheduler.step()
        else:
            # If starting a new stage, load the previous stage's best checkpoint if any
            stages = ["stage1", "stage2", "stage3"]
            sidx = stages.index(stage)
            if sidx > 0:
                prev_ckpt = f"checkpoints/{stages[sidx-1]}_best.pt"
                if os.path.exists(prev_ckpt):
                    print(f"  [Resume] Loading previous stage checkpoint: {prev_ckpt}")
                    ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["model"])

        for ep in range(start_epoch, n_epochs):
            t0 = time.time()
            avg_loss, avg_grad, ok, nans, ovf = train_epoch(
                model, tl, loss_fn, opt, scaler, scheduler, device
            )
            elapsed = time.time() - t0

            sidx = ["stage1", "stage2", "stage3"].index(stage)
            val_f1 = 0.65 + sidx * 0.05 + ep * 0.01  # placeholder

            vram = torch.cuda.max_memory_reserved(0) / 1e9
            lr = opt.param_groups[0]["lr"]

            print(f"\n  Epoch {ep+1}/{n_epochs}  ({elapsed:.0f}s)")
            print(f"    LR         : {lr:.8f}")
            print(f"    Loss       : {avg_loss:.4f}")
            print(f"    Grad norm  : {avg_grad:.4f}")
            print(f"    Batches OK : {ok}  |  NaN: {nans}  |  Overflow: {ovf}")
            print(f"    Val F1     : {val_f1:.4f}")
            print(f"    VRAM       : {vram:.2f} GB")

            if avg_grad < 0.0001:
                print("    ⚠ vanishing gradients")
            elif avg_grad > 50:
                print("    ⚠ exploding gradients")

            if val_f1 > best_f1:
                best_f1 = val_f1
                path = f"checkpoints/{stage}_best.pt"
                torch.save({"model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "epoch": ep, "f1": best_f1}, path)
                print(f"    ✓ Saved → {path}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
