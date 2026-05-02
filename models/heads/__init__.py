from .sil import SILHead
from .mil import GatedAttentionMIL


def get_head(classification, feature_dim, out_channel=1):
    if classification == 'sil':
        return SILHead(feature_dim, out_channel)
    elif classification == 'mil':
        return GatedAttentionMIL(feature_dim)
    raise ValueError(f'Unknown classification mode: {classification}')
