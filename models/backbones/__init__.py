from .uni import UNIBackbone
from .medclip import MedCLIPBackbone
from .sammed2d import SAMMed2DBackbone
from .biomedclip import BiomedCLIPBackbone


def get_backbone(model_name, model_cfg, device):
    if model_name == 'uni':
        return UNIBackbone(model_cfg['weight_path'], device)
    elif model_name == 'medclip':
        return MedCLIPBackbone(model_cfg['weight_path'], device)
    elif model_name == 'sammed2d':
        return SAMMed2DBackbone(model_cfg['weight_path'], device)
    elif model_name == 'biomedclip':
        return BiomedCLIPBackbone(model_cfg['hf_hub'], device)
    raise ValueError(f'Unknown model: {model_name}')
