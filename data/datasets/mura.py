import numpy as np
from PIL import Image
from .base import BaseRadiologyDataset


class MURADataset(BaseRadiologyDataset):
    def _load_gray(self, idx):
        path = self.data['data_path'][idx]
        img = Image.open(path).convert('L')
        return np.array(img)
