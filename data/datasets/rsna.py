import numpy as np
from pydicom import dcmread
from .base import BaseRadiologyDataset


class RSNADataset(BaseRadiologyDataset):
    def __init__(self, df, model_name, adaptation, dataset_cfg, is_mil=False):
        super().__init__(df, model_name, adaptation, dataset_cfg, is_mil)
        self.image_root = dataset_cfg['image_root']

    def _load_gray(self, idx):
        patient_id = self.data['patient_id'][idx]
        arr = dcmread(f'{self.image_root}/{patient_id}.dcm').pixel_array.astype(np.float32)
        denom = arr.max() - arr.min()
        arr = (arr - arr.min()) / (denom if denom > 0 else 1.0)
        return (arr * 255).astype(np.uint8)
