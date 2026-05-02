import torch.nn as nn

_HIDDEN_DIMS = {
    1024: [768, 384],
    512: [384, 128],
}


class SILHead(nn.Module):
    def __init__(self, feature_dim, out_channel=1):
        super().__init__()
        hidden = _HIDDEN_DIMS.get(feature_dim, [feature_dim // 2, feature_dim // 4])
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden[0]), nn.ReLU(), nn.Dropout(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(),
            nn.Linear(hidden[1], out_channel),
        )

    def forward(self, x):
        return self.fc(x)
