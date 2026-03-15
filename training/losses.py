import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits [batch, 2],  targets: long [batch]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0 - 1e-7)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def entropy_regularization(gate_vals):
    """Entropy of gate values, clamped to prevent NaN in fp16."""
    g = gate_vals.float().clamp(min=1e-6, max=1.0 - 1e-6)
    entropy = -g * torch.log(g) - (1.0 - g) * torch.log(1.0 - g)
    return entropy.mean()


def compute_class_weights(dataset, label_key, num_classes, device):
    """
    Compute inverse-frequency class weights from a dataset.

    Args:
        dataset     : PyTorch dataset or list of samples
        label_key   : key to access label in each sample dict
        num_classes : number of classes
        device      : torch device

    Returns:
        weights: FloatTensor [num_classes] normalized to sum to num_classes

    Usage:
        weights = compute_class_weights(train_dataset, "label_sentiment", 3, device)
        # Then pass to MultiTaskLoss config as "sentiment_class_weights"
    """
    counts = torch.zeros(num_classes)
    for sample in dataset:
        label = sample[label_key].item() if hasattr(sample[label_key], 'item') else sample[label_key]
        if 0 <= label < num_classes:
            counts[label] += 1

    counts = counts.clamp(min=1)                    # avoid division by zero
    weights = 1.0 / counts                          # inverse frequency
    weights = weights / weights.sum() * num_classes # normalize: sum = num_classes
    return weights.to(device)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with:
    - Masked task losses (m_t = 0 when label = -1)
    - Weighted CrossEntropy for sentiment (fixes class imbalance)
    - Focal Loss for harmful content (handles minority class)
    - Gate entropy regularization
    
    Changes from v1:
    - sentiment_loss now uses class_weights (inverse frequency)
    - fake_news_loss now uses class_weights (optional)
    - weights are computed from training set support counts
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.lambda_fake      = config.get("lambda_fake_news", 1.0)
        self.lambda_sentiment = config.get("lambda_sentiment", 0.8)
        self.lambda_harmful   = config.get("lambda_harmful", 1.2)

        # ── Sentiment class weights ─────────────────────────────────────────
        # Your support: [185, 412, 589] → positive underrepresented
        # Weights = inverse frequency, normalized
        # Default uses your observed support from training data
        sentiment_weights = config.get("sentiment_class_weights", None)
        if sentiment_weights is not None:
            w = torch.tensor(sentiment_weights, dtype=torch.float32)
        else:
            # Computed from your observed support: [185, 412, 589]
            support = torch.tensor([185.0, 412.0, 589.0])
            w = 1.0 / support
            w = w / w.sum() * 3          # normalize: sum = num_classes
        self.register_buffer("sentiment_weights", w)

        # ── Fake news class weights (optional) ─────────────────────────────
        fake_weights = config.get("fake_class_weights", None)
        if fake_weights is not None:
            fw = torch.tensor(fake_weights, dtype=torch.float32)
            self.register_buffer("fake_weights", fw)
        else:
            self.fake_weights = None

        # ── Loss functions ──────────────────────────────────────────────────
        # Sentiment: weighted CrossEntropy — fixes your 0.55 F1 problem
        self.sentiment_loss = nn.CrossEntropyLoss(
            weight=self.sentiment_weights,
            reduction='none'
        )

        # Fake news: optionally weighted CrossEntropy
        if self.fake_weights is not None:
            self.fake_news_loss = nn.CrossEntropyLoss(
                weight=self.fake_weights,
                reduction='none'
            )
        else:
            self.fake_news_loss = nn.CrossEntropyLoss(reduction='none')

        # Harmful: Focal Loss — handles minority hate class
        gamma = config.get("focal_loss_gamma", 2.0)
        self.harmful_loss = FocalLoss(gamma=gamma, reduction='none')

    def forward(self, logits_dict, targets_dict, gate_dict=None, gate_lambda=0.0):
        device = next(iter(logits_dict.values())).device
        total_loss = torch.tensor(0.0, device=device)
        details = {}

        # ── 1. Fake News ────────────────────────────────────────────────────
        if "labels_fake" in targets_dict:
            mask = (targets_dict["labels_fake"] >= 0).float()
            if mask.sum() > 0:
                safe_t = targets_dict["labels_fake"].clamp(min=0)

                # Move weights to correct device if needed
                if self.fake_weights is not None:
                    self.fake_news_loss.weight = self.fake_weights.to(device)

                l = self.fake_news_loss(logits_dict["logits_fake"], safe_t)
                l = (l * mask).sum() / mask.sum()
                total_loss = total_loss + self.lambda_fake * l
                details["loss_fake"] = l.item()

        # ── 2. Sentiment ────────────────────────────────────────────────────
        # KEY FIX: weighted loss for class imbalance [185, 412, 589]
        if "labels_sentiment" in targets_dict:
            mask = (targets_dict["labels_sentiment"] >= 0).float()
            if mask.sum() > 0:
                safe_t = targets_dict["labels_sentiment"].clamp(min=0)

                # Ensure weights are on correct device
                self.sentiment_loss.weight = self.sentiment_weights.to(device)

                l = self.sentiment_loss(logits_dict["logits_sentiment"], safe_t)
                l = (l * mask).sum() / mask.sum()
                total_loss = total_loss + self.lambda_sentiment * l
                details["loss_sentiment"] = l.item()

        # ── 3. Harmful ──────────────────────────────────────────────────────
        if "labels_harmful" in targets_dict:
            mask = (targets_dict["labels_harmful"] >= 0).float()
            if mask.sum() > 0:
                safe_t = targets_dict["labels_harmful"].clamp(min=0)
                l = self.harmful_loss(logits_dict["logits_harmful"], safe_t)
                if l.dim() > 0:
                    l = (l * mask).sum() / mask.sum()
                total_loss = total_loss + self.lambda_harmful * l
                details["loss_harmful"] = l.item()

        # ── 4. Gate entropy regularization ──────────────────────────────────
        if gate_dict is not None and gate_lambda > 0.0:
            reg_loss = torch.tensor(0.0, device=device)
            n = 0
            for key in ("token_gates", "task_gate", "modal_gate"):
                if key in gate_dict:
                    reg_loss = reg_loss + entropy_regularization(gate_dict[key])
                    n += 1
            if n > 0:
                reg_loss = reg_loss / n
                total_loss = total_loss + gate_lambda * reg_loss
                details["loss_gate_reg"] = reg_loss.item()

        details["loss_total"] = total_loss.item()
        return total_loss, details


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = 8

    logits = {
        "logits_fake":      torch.randn(b, 2).to(device),
        "logits_sentiment": torch.randn(b, 3).to(device),
        "logits_harmful":   torch.randn(b, 2).to(device),
    }
    targets = {
        "labels_fake":      torch.tensor([0, 1, -1, 1, 0, 1, -1, 0], dtype=torch.long).to(device),
        "labels_sentiment": torch.tensor([-1, 0, 2, 1, -1, 0, 1, 2], dtype=torch.long).to(device),
        "labels_harmful":   torch.tensor([1, 0, 0, -1, 1, 0, 1, -1], dtype=torch.long).to(device),
    }
    gates = {
        "token_gates": torch.rand(b, 128, 1).to(device),
        "task_gate":   torch.rand(b, 1).to(device),
        "modal_gate":  torch.rand(b, 1).to(device),
    }

    # ── Test 1: Default weights (computed from your support [185, 412, 589]) ──
    config = {
        "lambda_fake_news":  1.0,
        "lambda_sentiment":  0.8,
        "lambda_harmful":    1.2,
        "focal_loss_gamma":  2.0,
    }
    criterion = MultiTaskLoss(config, device=device).to(device)
    loss, details = criterion(logits, targets, gates, gate_lambda=0.1)
    assert not torch.isnan(loss), "NaN loss detected!"
    assert isinstance(loss, torch.Tensor)
    print("[OK] Test 1 — Default weights")
    print(f"     Sentiment weights: {criterion.sentiment_weights.tolist()}")
    for k, v in details.items():
        print(f"     {k}: {v:.4f}")

    # ── Test 2: Custom weights passed via config ──────────────────────────────
    config2 = {
        "lambda_fake_news":        1.0,
        "lambda_sentiment":        0.8,
        "lambda_harmful":          1.2,
        "focal_loss_gamma":        2.0,
        "sentiment_class_weights": [3.19, 1.43, 1.00],  # manually computed
    }
    criterion2 = MultiTaskLoss(config2, device=device).to(device)
    loss2, details2 = criterion2(logits, targets, gates, gate_lambda=0.1)
    assert not torch.isnan(loss2), "NaN loss detected!"
    print("\n[OK] Test 2 — Custom weights")
    print(f"     Sentiment weights: {criterion2.sentiment_weights.tolist()}")

    # ── Test 3: No NaN with extreme gate values ───────────────────────────────
    extreme_gates = {
        "token_gates": torch.zeros(b, 128, 1).to(device),   # all zeros
        "task_gate":   torch.ones(b, 1).to(device),          # all ones
        "modal_gate":  torch.rand(b, 1).to(device),
    }
    loss3, _ = criterion(logits, targets, extreme_gates, gate_lambda=0.1)
    assert not torch.isnan(loss3), "NaN with extreme gate values!"
    print("\n[OK] Test 3 — No NaN with extreme gate values (0 and 1)")

    print("\n✅ All loss tests passed.")