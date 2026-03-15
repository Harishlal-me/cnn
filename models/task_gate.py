import torch
import torch.nn as nn

class TaskGate(nn.Module):
    def __init__(self):
        super(TaskGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(512 + 512 + 64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_agg, cls_feat, task_emb):
        # h_agg: [batch, 512]
        # cls_feat: [batch, 512]
        # task_emb: [batch, 64]
        
        concat_feat = torch.cat([h_agg, cls_feat, task_emb], dim=-1) # [batch, 1088]
        gate_val = self.gate(concat_feat) # [batch, 1]
        
        fused = gate_val * h_agg + (1 - gate_val) * cls_feat # [batch, 512]
        return fused, gate_val

if __name__ == "__main__":
    h_agg = torch.randn(4, 512)
    cls_feat = torch.randn(4, 512)
    task_emb = torch.randn(4, 64)
    fused, gate = TaskGate()(h_agg, cls_feat, task_emb)
    assert fused.shape == (4, 512)
    assert gate.shape == (4, 1)
    print("[OK] TaskGate OK")
