import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob, is_binary=True):
    if len(y_true) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0}
    
    # metrics for binary and multi-class
    average_type = "binary" if is_binary else "macro"
    
    f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
    prec = precision_score(y_true, y_pred, average=average_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_type, zero_division=0)
    
    try:
        # AUC needs prob space appropriate to classes
        if is_binary:
            auc = roc_auc_score(y_true, y_prob[:, 1]) # assumes class 1 is positive
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except ValueError:
        # Can happen if only one class is present in y_true
        auc = 0.0
        
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(recall),
        "auc": float(auc)
    }

def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    
    all_preds_fake, all_labels_fake, all_probs_fake = [], [], []
    all_preds_sent, all_labels_sent, all_probs_sent = [], [], []
    all_preds_harm, all_labels_harm, all_probs_harm = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            has_image = batch["has_image"].to(device)
            task_id = batch["task_id"].to(device)
            
            out = model(input_ids, attention_mask, image, has_image, task_id)
            
            # Extract valid samples for Fake News
            mask_fake = batch["labels_fake"] != -1
            if mask_fake.sum() > 0:
                y_true = batch["labels_fake"][mask_fake].cpu().numpy()
                y_prob = torch.softmax(out["logits_fake"][mask_fake], dim=-1)
                y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
                all_labels_fake.extend(y_true)
                all_preds_fake.extend(y_pred)
                all_probs_fake.extend(y_prob.cpu().numpy())
                
            # Extract valid samples for Sentiment
            mask_sent = batch["labels_sentiment"] != -1
            if mask_sent.sum() > 0:
                y_true = batch["labels_sentiment"][mask_sent].cpu().numpy()
                y_prob = torch.softmax(out["logits_sentiment"][mask_sent], dim=-1)
                y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
                all_labels_sent.extend(y_true)
                all_preds_sent.extend(y_pred)
                all_probs_sent.extend(y_prob.cpu().numpy())
                
            # Extract valid samples for Harmful
            mask_harm = batch["labels_harmful"] != -1
            if mask_harm.sum() > 0:
                y_true = batch["labels_harmful"][mask_harm].cpu().numpy()
                y_prob = torch.softmax(out["logits_harmful"][mask_harm], dim=-1)
                y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
                all_labels_harm.extend(y_true)
                all_preds_harm.extend(y_pred)
                all_probs_harm.extend(y_prob.cpu().numpy())

    results = {
        "fake_news": calculate_metrics(np.array(all_labels_fake), np.array(all_preds_fake), np.array(all_probs_fake), is_binary=True),
        "sentiment": calculate_metrics(np.array(all_labels_sent), np.array(all_preds_sent), np.array(all_probs_sent), is_binary=False),
        "harmful": calculate_metrics(np.array(all_labels_harm), np.array(all_preds_harm), np.array(all_probs_harm), is_binary=True),
    }
    
    return results

if __name__ == "__main__":
    # Test Calculate Metrics
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_prob = np.array([[0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7], [0.9, 0.1]])
    res = calculate_metrics(y_true, y_pred, y_prob, is_binary=True)
    assert 'f1' in res and 'auc' in res
    print("[OK] Evaluate Metric functions OK")
