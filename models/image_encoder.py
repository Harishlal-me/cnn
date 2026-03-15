import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove final FC layer (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # out is [batch, 2048, 1, 1]
        self.proj = nn.Linear(2048, 512)

    def forward(self, image):
        # image: [batch, 3, 224, 224]
        feats = self.backbone(image) # [batch, 2048, 1, 1]
        feats = feats.view(feats.size(0), -1) # [batch, 2048]
        out = self.proj(feats) # [batch, 512]
        return out

if __name__ == "__main__":
    img = torch.randn(4, 3, 224, 224)
    out = ImageEncoder()(img)
    assert out.shape == (4, 512)
    print("[OK] ImageEncoder OK")
