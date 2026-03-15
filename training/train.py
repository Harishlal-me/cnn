import torch
import torch.nn as nn
from tqdm import tqdm
import os

from training.losses import MultiTaskLoss
from training.evaluate import evaluate_model
from models.mm_taskgate import MMTaskGate
import yaml

def train_stage(model, stage_config, train_loader, val_loader, loss_fn, device, stage_name):
    print(f"=== Starting {stage_name} ===")
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    opt_config = stage_config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_config), weight_decay=stage_config["weight_decay"])
    
    accum_steps = 4 # Default from user's global config
    gate_reg_lambda = stage_config.get("gate_reg_lambda", 0.0)
    
    # Apply Freezing based on phase config
    
    # 1. Image Encoder
    if stage_config.get("freeze_image_encoder", False):
        print("Freezing image encoder...")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    else:
        # Partial freezing
        freeze_layers = stage_config.get("freeze_resnet_layers", 0)
        print(f"Unfreezing image encoder. Freezing first {freeze_layers} layer groups.")
        for name, param in model.image_encoder.backbone.named_parameters():
             # Basic heuristic for Resnet layer freezing
             if freeze_layers >= 2 and ("layer1" in name or "layer2" in name):
                 param.requires_grad = False
             else:
                 param.requires_grad = True

    # 2. DistilBert Layer Freezing
    db_frozen = stage_config.get("freeze_distilbert_layers", 0)
    print(f"Freezing first {db_frozen} transformer layers...")
    for i in range(db_frozen):
        for param in model.transformer_branch.encoder.transformer.layer[i].parameters():
            param.requires_grad = False
            
    # 3. Cross Modal Gate
    if stage_config.get("freeze_cross_modal_gate", False):
        print("Freezing cross modal gate...")
        for param in model.cross_modal_gate.parameters():
            param.requires_grad = False
    else:
        print("Unfreezing cross modal gate...")
        for param in model.cross_modal_gate.parameters():
            param.requires_grad = True
            
    # Training Loop
    epochs = stage_config["epochs"]
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # We would loop over train_loader here
        # Mock training step for architectural structural demonstration
        
        # with torch.cuda.amp.autocast(enabled=True):
        #    out = model(...)
        #    loss, details = loss_fn(out, targets, gates, gate_lambda=gate_reg_lambda)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        print(f"[{stage_name}] Epoch {epoch+1}/{epochs} Completed.")
        
        # evaluate_model(model, val_loader, device=device)
        # log to wandb here

def run_curriculum(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
         config = {"stage1": {"epochs": 5, "learning_rate": 2.0e-5, "weight_decay": 0.01}, 
                   "stage2": {"epochs": 5, "learning_rate": 1e-5, "weight_decay": 0.01},
                   "stage3": {"epochs": 5, "learning_rate": 5e-6, "weight_decay": 0.01, "gate_reg_lambda": 0.1}}
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Init Model
    model = MMTaskGate(num_tasks=config.get("num_tasks", 3)).to(device)
    loss_fn = MultiTaskLoss(config)
    
    # Mock data loaders
    train_loader, val_loader = [], []

    # Run STAGE 1
    train_stage(model, config["stage1"], train_loader, val_loader, loss_fn, device, "Stage 1")
    
    # Run STAGE 2
    train_stage(model, config["stage2"], train_loader, val_loader, loss_fn, device, "Stage 2")
    
    # Run STAGE 3
    train_stage(model, config["stage3"], train_loader, val_loader, loss_fn, device, "Stage 3")

    return True

if __name__ == "__main__":
    print("[OK] Train file setup completed.")
