import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mm_taskgate import MMTaskGate
from data.mm_dataset import MMDataset
from training.losses import MultiTaskLoss

def sanity_test():
    print("Running Training Sanity Test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA GPU Required for Sanity Check.")
        return
        
    dataset = MMDataset("datasets/train.csv")
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    model = MMTaskGate(num_tasks=3).to(device)
    
    class ConfigMock:
         loss_weights = {"fake_news": 1.0, "sentiment": 1.0, "harmful": 1.0}
         def __getitem__(self, key):
             return self.loss_weights.get(key, 1.0)
         def get(self, key, default):
             return self.loss_weights.get(key, default)
             
    loss_fn = MultiTaskLoss(ConfigMock()).to(device)
    
    model.train()
    
    for i, batch in enumerate(loader):
        if i >= 3:
            break
            
        print(f"\n--- Batch {i+1} ---")
        print(f"Batch shape: Images {list(batch['image'].shape)}, Text {list(batch['input_ids'].shape)}")
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        has_image = batch["has_image"].to(device)
        task_id = batch["task_id"].to(device)
        label = batch["label"].to(device)
        
        labels_fake = torch.where(task_id == 0, label, torch.tensor(-1, dtype=torch.long, device=device))
        labels_sentiment = torch.where(task_id == 1, label, torch.tensor(-1, dtype=torch.long, device=device))
        labels_harmful = torch.where(task_id == 2, label, torch.tensor(-1, dtype=torch.long, device=device))
        
        targets_dict = {
            "labels_fake": labels_fake,
            "labels_sentiment": labels_sentiment,
            "labels_harmful": labels_harmful
        }
        
        with torch.cuda.amp.autocast(enabled=True):
            out = model(input_ids, attention_mask, image, has_image, task_id)
            gates = {
                "token_gates": out["token_gates"], 
                "task_gate": out["task_gate"], 
                "modal_gate": out["modal_gate"]
            }
            loss, details = loss_fn(out, targets_dict, gates, gate_lambda=0.01)
            
        print(f"Forward pass completed.")
        print(f"Loss value: {loss.item():.4f}")
        
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"\nFinal sanity GPU memory usage: {alloc:.3f} GB")
    print("Sanity Pass: SUCCESS")

if __name__ == "__main__":
    sanity_test()
