from .pathradx import PathRadX
from .backbones import get_backbone
from .heads import get_head


def build_model(model_name, adaptation, classification, model_cfg, device):
    backbone = get_backbone(model_name, model_cfg, device)
    backbone.freeze()
    head = get_head(classification, backbone.feature_dim)
    return PathRadX(backbone, adaptation, head).to(device)
