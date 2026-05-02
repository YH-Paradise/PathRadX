import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


class BiomedCLIPBackbone(nn.Module):
    feature_dim = 512
    input_size = 224

    def __init__(self, hf_hub, device):
        super().__init__()
        from open_clip import create_model_from_pretrained
        model, _ = create_model_from_pretrained(f'hf-hub:{hf_hub}')
        self.model = model.visual.to(device)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
