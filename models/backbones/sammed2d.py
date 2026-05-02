import torch
import torch.nn as nn
from argparse import Namespace


class SAMMed2DBackbone(nn.Module):
    feature_dim = 1024
    input_size = 256

    def __init__(self, weight_path, device):
        super().__init__()
        from models.segment_anything import sam_model_registry
        args = Namespace(image_size=256, encoder_adapter=True, sam_checkpoint=weight_path)
        sam = sam_model_registry['vit_b'](args).to(device)
        self.encoder = sam.image_encoder
        # Trainable projection: SAM outputs (B, 256, 64, 64) → flatten → (B, 1024)
        self.proj = nn.Sequential(
            nn.Linear(256 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Dropout(),
        ).to(device)

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        # proj remains trainable

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.proj(x)
