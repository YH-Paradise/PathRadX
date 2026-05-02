import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    def __init__(self, feature_dim=1024, attention_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: (B, P, feature_dim)
        A = self.attention_w(self.attention_V(x) * self.attention_U(x))  # (B, P, 1)
        A = F.softmax(A, dim=1)
        Z = torch.sum(A * x, dim=1)  # (B, feature_dim)
        logits = self.classifier(Z)  # (B, 1)
        return logits, A
