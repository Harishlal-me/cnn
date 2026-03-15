import torch
import torch.nn as nn

from models.multiscale_cnn import MultiScaleCNN
from models.transformer_branch import TransformerBranch
from models.token_gate import TokenGate
from models.task_gate import TaskGate
from models.image_encoder import ImageEncoder
from models.cross_modal_gate import CrossModalGate
from models.task_heads import TaskHeads

class MMTaskGate(nn.Module):
    def __init__(self, num_tasks=3):
        super(MMTaskGate, self).__init__()
        self.transformer_branch = TransformerBranch()
        self.multiscale_cnn = MultiScaleCNN()
        self.task_embedding = nn.Embedding(num_tasks, 64)
        
        self.token_gate = TokenGate()
        self.task_gate = TaskGate()
        self.image_encoder = ImageEncoder()
        self.cross_modal_gate = CrossModalGate()
        self.task_heads = TaskHeads()

    def forward(self, input_ids, attention_mask, image, has_image, task_id):
        # 1. Text branch features — RoBERTa embeddings for CNN branch
        embeddings = self.transformer_branch.encoder.embeddings(input_ids) # [batch, seq_len, 768]
        
        hidden, cls = self.transformer_branch(input_ids, attention_mask)
        cnn_feat = self.multiscale_cnn(embeddings)
        task_emb = self.task_embedding(task_id)

        # 2. Token gate
        token_fused, token_gates = self.token_gate(cnn_feat, hidden, task_emb)

        # 3. Mean pool + task gate
        h_agg = token_fused.mean(dim=1)
        text_feat, task_gate_val = self.task_gate(h_agg, cls, task_emb)

        # 4. Image branch
        visual_feat = self.image_encoder(image)

        # 5. Cross-modal gate
        fused, modal_gate_val = self.cross_modal_gate(text_feat, visual_feat, has_image)

        # 6. Task heads
        logits_fake, logits_sentiment, logits_harmful = self.task_heads(fused)

        return {
            "logits_fake": logits_fake,
            "logits_sentiment": logits_sentiment,
            "logits_harmful": logits_harmful,
            "token_gates": token_gates,
            "task_gate": task_gate_val,
            "modal_gate": modal_gate_val,
        }

if __name__ == "__main__":
    model = MMTaskGate()
    batch = {
        "input_ids":       torch.randint(0, 1000, (4, 128)),
        "attention_mask":  torch.ones(4, 128, dtype=torch.long),
        "image":           torch.randn(4, 3, 224, 224),
        "has_image":       torch.tensor([True, True, False, False]),
        "task_id":         torch.tensor([0, 1, 2, 0]),
    }
    out = model(**batch)
    assert out["logits_fake"].shape      == (4, 2)
    assert out["logits_sentiment"].shape == (4, 3)
    assert out["logits_harmful"].shape   == (4, 2)
    print("[OK] Full MMTaskGate forward pass OK")
