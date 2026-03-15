import torch
import torch.nn as nn

class CrossModalGate(nn.Module):
    def __init__(self):
        super(CrossModalGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(512 + 512, 1),
            nn.Sigmoid()
        )

    def forward(self, text_feat, visual_feat, has_image):
        # text_feat: [batch, 512]
        # visual_feat: [batch, 512]
        # has_image: [batch] boolean mask
        
        concat_feat = torch.cat([text_feat, visual_feat], dim=-1)
        g_modal = self.gate(concat_feat) # [batch, 1]
        
        # When has_image=False, gate_val = 1.0
        # g_modal shape is [batch, 1], has_image is [batch]
        has_image_mask = has_image.unsqueeze(1).float() # [batch, 1]
        
        g_modal_final = g_modal * has_image_mask + (1.0 - has_image_mask)
        fused = g_modal_final * text_feat + (1 - g_modal_final) * visual_feat
        
        return fused, g_modal_final

if __name__ == "__main__":
    text_feat = torch.randn(4, 512)
    visual_feat = torch.randn(4, 512)
    has_image = torch.tensor([True, True, False, False])
    fused, gates = CrossModalGate()(text_feat, visual_feat, has_image)
    assert fused.shape == (4, 512)
    assert gates.shape == (4, 1)
    # Check bypass
    assert gates[2].item() == 1.0
    assert gates[3].item() == 1.0
    print("[OK] CrossModalGate OK")
