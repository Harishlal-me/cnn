import torch
import torch.nn as nn
from transformers import RobertaModel


class TransformerBranch(nn.Module):
    """Text backbone — RoBERTa-base (768-dim → projected to 512)."""

    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.encoder.gradient_checkpointing_enable()
        self.proj_hidden = nn.Linear(768, 512)
        self.proj_cls = nn.Linear(768, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state   # [B, seq, 768]
        cls_token = hidden_states[:, 0, :]           # [B, 768]

        hidden_proj = self.proj_hidden(hidden_states)  # [B, seq, 512]
        cls_proj = self.proj_cls(cls_token)             # [B, 512]
        return hidden_proj, cls_proj


if __name__ == "__main__":
    input_ids = torch.randint(0, 1000, (4, 128))
    mask = torch.ones(4, 128, dtype=torch.long)
    hidden, cls = TransformerBranch()(input_ids, mask)
    assert hidden.shape == (4, 128, 512)
    assert cls.shape == (4, 512)
    print("[OK] TransformerBranch (RoBERTa) OK")
