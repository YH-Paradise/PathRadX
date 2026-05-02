import torch
import torch.nn as nn
from einops import rearrange
from .heads.mil import GatedAttentionMIL


class PathRadX(nn.Module):
    def __init__(self, backbone, adaptation, head):
        """
        backbone : a backbone instance (UNIBackbone, SAMMed2DBackbone, etc.)
        adaptation : 'cm' (Conv2d 1→3) or 'pc' (Identity, input is already RGB)
        head : SILHead or GatedAttentionMIL
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.is_mil = isinstance(head, GatedAttentionMIL)

        if adaptation == 'cm':
            self.adaptation_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.adaptation_layer = nn.Identity()

    def forward(self, x):
        if self.is_mil:
            B, P = x.shape[:2]
            x = rearrange(x, 'b p c h w -> (b p) c h w')
            x = self.adaptation_layer(x)
            x = self.backbone(x)
            x = rearrange(x, '(b p) t -> b p t', b=B)
            return self.head(x)  # (logits, attention)
        else:
            x = self.adaptation_layer(x)
            x = self.backbone(x)
            return self.head(x)  # logits
