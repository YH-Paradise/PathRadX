import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from data.datasets import get_dataloader
from models import build_model
from training.losses import get_loss
from training.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='PathRadX training')
    parser.add_argument('--dataset', type=str, required=True, choices=['mura', 'rsna', 'medfmc'])
    parser.add_argument('--model', type=str, required=True, choices=['uni', 'medclip', 'sammed2d', 'biomedclip'])
    parser.add_argument('--adaptation', type=str, default='cm', choices=['cm', 'pc'],
                        help='cm: Channel Manipulation, pc: Pseudo Coloring')
    parser.add_argument('--classification', type=str, default='sil', choices=['sil', 'mil'],
                        help='sil: Single Instance Learning, mil: Multiple Instance Learning')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging (requires WANDB_API_KEY env var)')
    args = parser.parse_args()

    set_seed()
    device = f'cuda:{args.cuda_num}' if torch.cuda.is_available() else 'cpu'

    dataset_cfg = load_yaml(f'configs/datasets/{args.dataset}.yaml')
    model_cfg = load_yaml(f'configs/models/{args.model}.yaml')

    csv_dir = dataset_cfg['csv_dir']
    train_df = pd.read_csv(os.path.join(csv_dir, dataset_cfg['csv_files']['train']))
    val_df = pd.read_csv(os.path.join(csv_dir, dataset_cfg['csv_files']['val']))

    is_mil = args.classification == 'mil'
    train_loader = get_dataloader(
        args.dataset, train_df, args.model, args.adaptation,
        is_mil=is_mil, dataset_cfg=dataset_cfg, batch_size=args.batch_size, shuffle=True,
    )
    val_loader = get_dataloader(
        args.dataset, val_df, args.model, args.adaptation,
        is_mil=is_mil, dataset_cfg=dataset_cfg, batch_size=args.batch_size, shuffle=False,
    )

    model = build_model(args.model, args.adaptation, args.classification, model_cfg, device)

    loss_fn = get_loss('focal' if is_mil else 'bce')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=10, epochs=10, anneal_strategy='cos'
    )

    exp_name = f'{args.model}_{args.dataset}_{args.adaptation}_{args.classification}'
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project='PathRadX', name=exp_name)

    trainer = Trainer(model, loss_fn, optimizer, scheduler, device, is_mil, save_dir, args.wandb)
    trainer.fit(train_loader, val_loader, args.epochs)


if __name__ == '__main__':
    main()
