import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class MedCLIPBackbone(nn.Module):
    feature_dim = 512
    input_size = 224

    def __init__(self, weight_path, device):
        super().__init__()
        from medclip import MedCLIPModel, MedCLIPVisionModelViT
        medclip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT).to(device)
        # The MedCLIP library initializes weights internally; the external .bin file
        # contains the full model state — only the vision encoder portion is used here.
        if weight_path:
            state = torch.load(weight_path, map_location=device, weights_only=True)
            vision_state = {k.replace('vision_model.', ''): v for k, v in state.items() if k.startswith('vision_model.')}
            if vision_state:
                medclip.vision_model.load_state_dict(vision_state, strict=False)
        self.model = medclip.vision_model

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
