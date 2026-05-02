import numpy as np
from PIL import Image
from .base import BaseRadiologyDataset


class MedFMCDataset(BaseRadiologyDataset):
    def _load_gray(self, idx):
        path = self.data['img_dir'][idx]
        img = Image.open(path).convert('L')
        return np.array(img)
