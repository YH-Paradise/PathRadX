import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


def apply_pseudo_coloring(gray_array):
    cmap = plt.get_cmap('RdPu')
    arr = gray_array.astype(np.float32)
    denom = arr.max() - arr.min()
    arr = (arr - arr.min()) / (denom if denom > 0 else 1.0)
    colored = cmap(arr)
    return Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))


def _normalize_patch(patch_img):
    return transforms.Normalize([0.449], [0.226])(transforms.ToTensor()(patch_img))


def get_sil_transform(model_name):
    out_size = 256 if model_name == 'sammed2d' else 224
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((out_size, out_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.449], [0.226]),
    ])


def get_mil_resize(dataset_cfg):
    resize_to = dataset_cfg.get('mil', {}).get('resize_to')
    if resize_to:
        return transforms.Resize((resize_to, resize_to))
    return None


class BaseRadiologyDataset(Dataset):
    def __init__(self, df, model_name, adaptation, dataset_cfg, is_mil=False):
        self.data = df.reset_index(drop=True)
        self.model_name = model_name
        self.adaptation = adaptation
        self.is_mil = is_mil
        self.label_col = dataset_cfg.get('label_col', 'label')

        mil_cfg = dataset_cfg.get('mil', {})
        self.patch_size = mil_cfg.get('patch_size', 224)
        overlaps = mil_cfg.get('overlap', {})
        self.overlap = overlaps.get(model_name, overlaps.get('default', 0))

        self.sil_transform = get_sil_transform(model_name)
        self.mil_resize = get_mil_resize(dataset_cfg)

    def _load_gray(self, idx) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def _extract_patches(self, arr):
        step = self.patch_size - self.overlap
        h, w = arr.shape[:2]
        patches = []
        for i in range(0, h - self.patch_size + 1, step):
            for j in range(0, w - self.patch_size + 1, step):
                patches.append(Image.fromarray(arr[i:i + self.patch_size, j:j + self.patch_size]))
        if not patches:
            patches.append(Image.fromarray(arr).resize((self.patch_size, self.patch_size)))
        return patches

    def __getitem__(self, idx):
        gray = self._load_gray(idx)
        label = torch.tensor(int(self.data[self.label_col][idx]), dtype=torch.float32)

        if self.adaptation == 'pc':
            img = apply_pseudo_coloring(gray)
        else:
            img = Image.fromarray(gray.astype(np.uint8)).convert('L')

        if self.is_mil:
            if self.mil_resize is not None:
                img = self.mil_resize(img)
            arr = np.array(img)
            patches = self._extract_patches(arr)
            patch_tensors = torch.stack([_normalize_patch(p) for p in patches])
            return patch_tensors, label
        else:
            return self.sil_transform(img), label
