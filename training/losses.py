import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._alpha_tensor = None

    def _get_alpha(self, device):
        if self._alpha_tensor is None or self._alpha_tensor.device != device:
            self._alpha_tensor = torch.tensor([self.alpha, 1 - self.alpha], device=device)
        return self._alpha_tensor

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        at = self._get_alpha(inputs.device).gather(0, targets.long().view(-1))
        return (at * (1 - pt) ** self.gamma * bce).mean()


def get_loss(loss_type='bce'):
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        return WeightedFocalLoss()
    raise ValueError(f'Unknown loss type: {loss_type}')
