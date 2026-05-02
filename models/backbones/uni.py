import torch
import torch.nn as nn
import timm


class UNIBackbone(nn.Module):
    feature_dim = 1024
    input_size = 224

    def __init__(self, weight_path, device):
        super().__init__()
        self.model = timm.create_model(
            'vit_large_patch16_224',
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
        ).to(device)
        state = torch.load(weight_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state, strict=True)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
