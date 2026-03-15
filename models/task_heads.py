import torch
import torch.nn as nn

class TaskHeads(nn.Module):
    def __init__(self):
        super(TaskHeads, self).__init__()
        self.head_fake = nn.Linear(512, 2)
        self.head_sentiment = nn.Linear(512, 3)
        self.head_harmful = nn.Linear(512, 2)

    def forward(self, shared_repr):
        logits_fake = self.head_fake(shared_repr)
        logits_sentiment = self.head_sentiment(shared_repr)
        logits_harmful = self.head_harmful(shared_repr)
        return logits_fake, logits_sentiment, logits_harmful

if __name__ == "__main__":
    shared_repr = torch.randn(4, 512)
    logits_fake, logits_sentiment, logits_harmful = TaskHeads()(shared_repr)
    assert logits_fake.shape == (4, 2)
    assert logits_sentiment.shape == (4, 3)
    assert logits_harmful.shape == (4, 2)
    print("[OK] TaskHeads OK")
