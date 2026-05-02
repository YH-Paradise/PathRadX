from torch.utils.data import DataLoader
from .mura import MURADataset
from .rsna import RSNADataset
from .medfmc import MedFMCDataset

_DATASET_MAP = {
    'mura': MURADataset,
    'rsna': RSNADataset,
    'medfmc': MedFMCDataset,
}


def get_dataloader(dataset_name, df, model_name, adaptation, is_mil, dataset_cfg, batch_size, shuffle):
    cls = _DATASET_MAP[dataset_name]
    dataset = cls(df, model_name=model_name, adaptation=adaptation, dataset_cfg=dataset_cfg, is_mil=is_mil)
    num_workers = 32 if is_mil else 8
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
