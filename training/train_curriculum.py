import os
import yaml
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.mm_taskgate import MMTaskGate
from training.losses import MultiTaskLoss
from training.evaluate import evaluate_model
from Datasets import build_dataloader

def get_optimizer_for_stage(model, config, stage):
    lr = float(config[stage]["learning_rate"])
    weight_decay = float(config[stage]["weight_decay"])
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, 
                           weight_decay=weight_decay)
    return optimizer

def run_epoch(model, dataloader, optimizer, scaler, loss_fn, gate_reg_lambda, device, accum_steps=4):
    model.train()
    total_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        has_image = batch["has_image"].to(device)
        
        # Determine active task based on valid labels (non -1)
        task_id = torch.zeros(input_ids.size(0), dtype=torch.long, device=device)
        task_id = torch.where(batch["label_sentiment"].to(device) != -1, torch.tensor(1, device=device), task_id)
        task_id = torch.where(batch["label_harmful"].to(device) != -1, torch.tensor(2, device=device), task_id)
        
        targets_dict = {
            "labels_fake": batch["label_fake"].to(device),
            "labels_sentiment": batch["label_sentiment"].to(device),
            "labels_harmful": batch["label_harmful"].to(device)
        }
        
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(input_ids, attention_mask, image, has_image, task_id)
            gates = {
                "token_gates": out["token_gates"], 
                "task_gate": out["task_gate"], 
                "modal_gate": out["modal_gate"]
            }
            loss, details = loss_fn(out, targets_dict, gates, gate_lambda=gate_reg_lambda)
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += details["loss_total"]
        
    return total_loss / len(dataloader)


def build_curriculum(config_path="config.yaml"):
    print("Loading curriculum config...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # STEP 1: Device detection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
        print("Using device: cuda:0")
    else:
        print("Using device: cpu")
        
    model = MMTaskGate(num_tasks=int(config["num_tasks"]))
    model = model.to(device)
    loss_fn = MultiTaskLoss(config).to(device)
    
    # STEP 2: Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Real Dataloaders - This ensures the run takes the proper ~7 hours
    data_root = config.get("data_root", "./datasets")
    train_loader = build_dataloader(data_root, split="train", batch_size=config["stage1"]["batch_size"], num_workers=config.get("num_workers", 2))
    val_loader = build_dataloader(data_root, split="val", batch_size=config["stage1"]["batch_size"], num_workers=config.get("num_workers", 2))

    report_path = "results/phase3_gpu_training_result.txt"
    with open(report_path, "w") as f:
        f.write("Phase 3 - Curriculum GPU Training Execution Report\n")
        if device.type == "cuda":
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB\n")
        f.write("="*50 + "\n\n")

    for stage_idx, stage_name in enumerate(["stage1", "stage2", "stage3"]):
        stage_cfg = config[stage_name]
        epochs = stage_cfg['epochs']
        gate_reg_lambda = stage_cfg.get('gate_reg_lambda', 0.0)
        accum_steps = config.get("gradient_accumulation_steps", 4)
        
        # Adjust dataloader batch size if it changes per stage
        train_loader = build_dataloader(data_root, split="train", batch_size=stage_cfg["batch_size"], num_workers=config.get("num_workers", 2))
        val_loader = build_dataloader(data_root, split="val", batch_size=stage_cfg["batch_size"], num_workers=config.get("num_workers", 2))
        
        if stage_idx > 0:
            prev_stage = f"stage{stage_idx}"
            ckpt_path = f"checkpoints/{prev_stage}_best_gpu.pt"
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint from {ckpt_path}...")
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])

        print(f"\n--- Starting {stage_name} Training ---")
        
        # Enable/Disable component tracking per phase rules
        if stage_cfg.get("freeze_image_encoder", False):
             for param in model.image_encoder.parameters():
                 param.requires_grad = False
        else:
             freeze_layers = stage_cfg.get("freeze_resnet_layers", 0)
             for name, param in model.image_encoder.backbone.named_parameters():
                 if freeze_layers >= 2 and ("layer1" in name or "layer2" in name):
                     param.requires_grad = False
                 else:
                     param.requires_grad = True

        if stage_cfg.get("freeze_cross_modal_gate", False):
             for param in model.cross_modal_gate.parameters():
                 param.requires_grad = False
        else:
             for param in model.cross_modal_gate.parameters():
                 param.requires_grad = True
                 
        db_frozen = stage_cfg.get("freeze_distilbert_layers", 0)
        for i in range(model.transformer_branch.encoder.config.num_hidden_layers):
             requires_grad = (i >= db_frozen)
             for param in model.transformer_branch.encoder.transformer.layer[i].parameters():
                 param.requires_grad = requires_grad
                 
        optimizer = get_optimizer_for_stage(model, config, stage_name)
        
        # STEP 3: Verify gradients for each stage
        print(f"[{stage_name.upper()}] Trainable Parameters Verification:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("TRAINABLE:", name)

        best_f1 = -1.0
        
        for epoch in range(epochs):
            # Train
            train_loss = run_epoch(model, train_loader, optimizer, scaler, loss_fn, gate_reg_lambda, device, accum_steps)
            
            # STEP 6: Check image branch gradients
            if not stage_cfg.get("freeze_image_encoder", False):
                print(f"[{stage_name.upper()}] Checking Image Branch Gradients:")
                for name, param in model.image_encoder.named_parameters():
                    if param.requires_grad:
                        print(name, param.grad is not None)
            
            # Formulate val loader
            class WrappedValLoader:
                def __init__(self, loader):
                    self.loader = loader
                def __iter__(self):
                    for b in self.loader:
                        t_id = torch.zeros(b["input_ids"].size(0), dtype=torch.long)
                        t_id = torch.where(b["label_sentiment"] != -1, torch.tensor(1), t_id)
                        t_id = torch.where(b["label_harmful"] != -1, torch.tensor(2), t_id)
                        b["task_id"] = t_id
                        b["labels_fake"] = b["label_fake"]
                        b["labels_sentiment"] = b["label_sentiment"]
                        b["labels_harmful"] = b["label_harmful"]
                        yield b
                        
            # Eval
            eval_results = evaluate_model(model, WrappedValLoader(val_loader), device=device)
            val_f1 = (eval_results["fake_news"]["f1"] + eval_results["sentiment"]["f1"] + eval_results["harmful"]["f1"]) / 3.0
            
            # STEP 5: GPU memory monitoring
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                max_alloc = torch.cuda.max_memory_allocated() / 1e9
                print(f"GPU allocated: {alloc:.3f} GB")
                print(f"GPU reserved: {resv:.3f} GB")
                
            # STEP 4: Ensure real training occurs
            log_str = f"Epoch {epoch+1}/{epochs}\nTrain Loss: {train_loss:.4f}\nValidation F1: {val_f1:.4f}"
            print(log_str)
            
            with open(report_path, "a") as f:
                f.write(f"[{stage_name.upper()}] {log_str}\n")
                if device.type == "cuda":
                    f.write(f"GPU allocated: {alloc:.3f} GB\n")
                
            # STEP 7: Save checkpoints correctly
            if val_f1 >= best_f1:
                best_f1 = val_f1
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1
                }
                torch.save(ckpt, f"checkpoints/{stage_name}_best_gpu.pt")
                
        with open(report_path, "a") as f:
            f.write(f"-> Saved {stage_name}_best_gpu.pt with Best F1: {best_f1:.4f}\n\n")

if __name__ == "__main__":
    build_curriculum()
