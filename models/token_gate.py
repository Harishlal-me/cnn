import torch
import torch.nn as nn

class TokenGate(nn.Module):
    def __init__(self):
        super(TokenGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(512 + 512 + 64, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, tf_feat, task_emb):
        # cnn_feat: [batch, seq_len, 512]
        # tf_feat: [batch, seq_len, 512]
        # task_emb: [batch, 64]
        
        seq_len = cnn_feat.size(1)
        task_emb_exp = task_emb.unsqueeze(1).expand(-1, seq_len, -1) # [batch, seq_len, 64]
        
        concat_feat = torch.cat([cnn_feat, tf_feat, task_emb_exp], dim=-1) # [batch, seq_len, 1088]
        gate_vals = self.gate(concat_feat) # [batch, seq_len, 1]
        
        fused = gate_vals * cnn_feat + (1 - gate_vals) * tf_feat # [batch, seq_len, 512]
        return fused, gate_vals

if __name__ == "__main__":
    cnn = torch.randn(4, 128, 512)
    tf = torch.randn(4, 128, 512)
    task = torch.randn(4, 64)
    fused, gates = TokenGate()(cnn, tf, task)
    assert fused.shape == (4, 128, 512)
    assert gates.shape == (4, 128, 1)
    assert gates.min() >= 0 and gates.max() <= 1
    print("[OK] TokenGate OK")
