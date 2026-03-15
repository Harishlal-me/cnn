import torch
import torch.nn as nn

class MultiScaleCNN(nn.Module):
    def __init__(self, embed_dim=768):
        super(MultiScaleCNN, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.proj = nn.Linear(384, 512)

    def forward(self, x):
        # Input: [batch, seq_len, embed_dim]
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        x_in = x.transpose(1, 2)
        
        c3 = self.conv3(x_in).transpose(1, 2) # [batch, seq_len, 128]
        c5 = self.conv5(x_in).transpose(1, 2)
        c7 = self.conv7(x_in).transpose(1, 2)
        
        # Concat along feature dim
        concat = torch.cat([c3, c5, c7], dim=2) # [batch, seq_len, 384]
        out = self.proj(concat) # [batch, seq_len, 512]
        return out

if __name__ == "__main__":
    x = torch.randn(4, 128, 768)   # batch=4, seq_len=128, embed=768
    out = MultiScaleCNN()(x)
    assert out.shape == (4, 128, 512)
    print("[OK] MultiScaleCNN OK")
