import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data.datasets import get_dataloader
from models import build_model
from training.metrics import compute_metrics, print_metrics


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device, is_mil):
    model.eval()
    model.to(device)

    all_probs, all_preds, all_gts = [], [], []

    with torch.no_grad():
        for data in tqdm(loader, ascii=True):
            x, y = data[0].to(device), data[1].to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_gts.extend(y.int().cpu().numpy().tolist())

    return compute_metrics(all_gts, all_preds, all_probs)


def main():
    parser = argparse.ArgumentParser(description='PathRadX evaluation')
    parser.add_argument('--dataset', type=str, required=True, choices=['mura', 'rsna', 'medfmc'])
    parser.add_argument('--model', type=str, required=True, choices=['uni', 'medclip', 'sammed2d', 'biomedclip'])
    parser.add_argument('--adaptation', type=str, default='cm', choices=['cm', 'pc'])
    parser.add_argument('--classification', type=str, default='sil', choices=['sil', 'mil'])
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cuda_num', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.cuda_num}' if torch.cuda.is_available() else 'cpu'

    dataset_cfg = load_yaml(f'configs/datasets/{args.dataset}.yaml')
    model_cfg = load_yaml(f'configs/models/{args.model}.yaml')

    csv_dir = dataset_cfg['csv_dir']
    test_df = pd.read_csv(os.path.join(csv_dir, dataset_cfg['csv_files']['test']))

    is_mil = args.classification == 'mil'
    test_loader = get_dataloader(
        args.dataset, test_df, args.model, args.adaptation,
        is_mil=is_mil, dataset_cfg=dataset_cfg, batch_size=args.batch_size, shuffle=False,
    )

    model = build_model(args.model, args.adaptation, args.classification, model_cfg, device)
    state = torch.load(args.weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)

    print(f'Model: {args.model} | Dataset: {args.dataset} | Adaptation: {args.adaptation} | Classification: {args.classification}')
    metrics = evaluate(model, test_loader, device, is_mil)
    print_metrics(metrics)


if __name__ == '__main__':
    main()
