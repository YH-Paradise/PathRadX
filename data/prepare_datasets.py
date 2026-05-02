"""
Prepare train/val/test CSV splits for each dataset.

Usage:
    python data/prepare_datasets.py --dataset_name MURA --dataset_dir /path/to/MURA-v1.1 --output_dir csvs
    python data/prepare_datasets.py --dataset_name RSNA --dataset_dir /path/to/rsna-challenge --output_dir csvs
    python data/prepare_datasets.py --dataset_name MedFMC --dataset_dir /path/to/MedFMC --output_dir csvs
"""

import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_mura(dataset_dir, output_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')

    train_imgs = glob.glob(os.path.join(train_dir, '*/*/*/*.png'))
    valid_imgs = glob.glob(os.path.join(valid_dir, '*/*/*/*.png'))

    def label_from_path(paths):
        return [1 if 'positive' in p.lower() else 0 for p in paths]

    train_df = pd.DataFrame({'data_path': train_imgs, 'label': label_from_path(train_imgs)})
    val_df = pd.DataFrame({'data_path': valid_imgs, 'label': label_from_path(valid_imgs)})

    # Create a held-out test split from the original validation set
    val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df['label'], random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'MURA_train_data_label_per_img.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'MURA_val_data_label_per_img.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'MURA_test_data_label_per_img.csv'), index=False)
    print(f'MURA: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')


def prepare_rsna(dataset_dir, output_dir):
    label_csv = os.path.join(dataset_dir, 'stage_2_train_labels.csv')
    raw = pd.read_csv(label_csv)

    # Deduplicate: one label per patient
    patients = raw.groupby('patientId')['Target'].first().reset_index()
    patients.columns = ['patient_id', 'label']

    pos = patients[patients['label'] == 1]
    neg = patients[patients['label'] == 0]

    def split_stratified(df):
        train, rest = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(rest, test_size=0.5, random_state=42)
        return train, val, test

    tr_p, va_p, te_p = split_stratified(pos)
    tr_n, va_n, te_n = split_stratified(neg)

    train_df = pd.concat([tr_p, tr_n]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([va_p, va_n]).reset_index(drop=True)
    test_df = pd.concat([te_p, te_n]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'rsna_pneumo_trainset_split(1204).csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'rsna_pneumo_valset_split(1204).csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'rsna_pneumo_testset_split(1204).csv'), index=False)
    print(f'RSNA: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')


def prepare_medfmc(dataset_dir, output_dir):
    annotation_csv = os.path.join(dataset_dir, 'MedFMC_trainval_annotation', 'chestdr_published_19d.csv')
    label_df = pd.read_csv(annotation_csv)

    train_imgs = sorted(glob.glob(os.path.join(dataset_dir, 'MedFMC_train', 'chest', 'images', '*')))
    val_imgs = sorted(glob.glob(os.path.join(dataset_dir, 'MedFMC_val', 'chest', 'images', '*')))
    all_imgs = train_imgs + val_imgs

    img_dirs, labels = [], []
    for path in all_imgs:
        filename = os.path.basename(path)
        row = label_df[label_df['img_id'] == filename]
        if row.empty:
            continue
        img_dirs.append(path)
        labels.append(1 if row.iloc[:, 2:21].values.sum() > 0 else 0)

    df = pd.DataFrame({'img_dir': img_dirs, 'label': labels})
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]

    def split_stratified(d):
        tr, rest = train_test_split(d, test_size=0.2, random_state=42)
        va, te = train_test_split(rest, test_size=0.5, random_state=42)
        return tr, va, te

    tr_p, va_p, te_p = split_stratified(pos)
    tr_n, va_n, te_n = split_stratified(neg)

    train_df = pd.concat([tr_p, tr_n]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([va_p, va_n]).reset_index(drop=True)
    test_df = pd.concat([te_p, te_n]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'MedFMC_chest_trainset_split(0211).csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'MedFMC_chest_valset_split(0211).csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'MedFMC_chest_testset_split(0211).csv'), index=False)
    print(f'MedFMC: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['MURA', 'RSNA', 'MedFMC'])
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory of the downloaded dataset')
    parser.add_argument('--output_dir', type=str, default='csvs')
    args = parser.parse_args()

    fn = {'MURA': prepare_mura, 'RSNA': prepare_rsna, 'MedFMC': prepare_medfmc}[args.dataset_name]
    fn(args.dataset_dir, args.output_dir)
