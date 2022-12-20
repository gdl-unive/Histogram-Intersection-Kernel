import argparse
import json
import os
import sys
from argparse import Namespace
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split

import wandb

sys.path.insert(0, './src')
from dataset import EgonetLoader, OgbDataset, TUDataset, ZINCDataset
from model import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_SILENT"] = "true"
DEVICE = 'cpu'
GPUS = 1


def prepare_tud_dataset(run_params):
    dataset = TUDataset('./data', run_params.dataset, hops=run_params.hops)
    yy = [int(d.y) for d in dataset]
    fold = run_params.fold

    # Load or generate splits
    folds_path = f'./data/folds/{run_params.dataset}_folds_{run_params.folds}.txt'
    if not os.path.isfile(folds_path):
        print(f'GENERATING {run_params.folds} FOLDS FOR {run_params.dataset}')
        skf = StratifiedKFold(n_splits=run_params.folds, random_state=1, shuffle=True)
        folds = list(skf.split(np.arange(len(yy)), yy))

        folds_split = []
        for fold in range(run_params.folds):
            train_i_split, val_i_split = train_test_split([int(i) for i in folds[fold][0]],
                                                          stratify=[n for n in np.asarray(yy)[folds[fold][0]]],
                                                          test_size=int(len(list(folds[fold][0])) * 0.1),
                                                          random_state=0)
            test_i_split = [int(i) for i in folds[fold][1]]
            folds_split.append([train_i_split, val_i_split, test_i_split])

        with open(folds_path, 'w') as f:
            f.write(json.dumps(folds_split))

    fold = run_params.fold
    with open(folds_path, 'r') as f:
        folds = json.loads(f.read())
    train_i_split, val_i_split, test_i_split = folds[fold]
    return dataset[train_i_split], dataset[val_i_split], dataset[test_i_split]


def prepare_zinc_dataset(run_params):
    train_dataset = ZINCDataset('./data', split='train', hops=run_params.hops)
    val_dataset = ZINCDataset('./data', split='val', hops=run_params.hops)
    test_dataset = ZINCDataset('./data', split='test', hops=run_params.hops)

    run_params.loss_func = torch.nn.L1Loss()
    run_params.num_classes = 1
    return train_dataset, val_dataset, test_dataset


def prepare_ogb_dataset(run_params):
    dataset = OgbDataset('ogbg-molhiv', hops=run_params.hops)
    split_idx = dataset.get_idx_split()
    return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]


def run_training_process_with_validation(run_params):
    print(f'''{"#" * 30} NEW TRAIN on FOLD {run_params.fold} {"#" * 30}''')
    train_dataset = None
    val_dataset = None
    test_dataset = None
    run_params.loss_func = torch.nn.CrossEntropyLoss()
    if run_params.dataset.upper() == 'ZINC':
        train_dataset, val_dataset, test_dataset = prepare_zinc_dataset(run_params)
    elif run_params.dataset.upper() == 'OGB':
        train_dataset, val_dataset, test_dataset = prepare_ogb_dataset(run_params)
    else:
        train_dataset, val_dataset, test_dataset = prepare_tud_dataset(run_params)

    run_params.in_features = train_dataset.num_features
    run_params.num_classes = train_dataset.num_classes

    train_loader = EgonetLoader(train_dataset, batch_size=run_params.batch_size, shuffle=True)
    val_loader = EgonetLoader(val_dataset, batch_size=run_params.batch_size, shuffle=True)
    test_loader = EgonetLoader(test_dataset, batch_size=run_params.batch_size, shuffle=True)

    class MyDataModule(pl.LightningDataModule):
        def setup(self, stage=None):
            pass

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return test_loader

    run_params.labels = 1

    model = Model(run_params, DEVICE)
    if DEVICE == 'cuda':
        model = model.cuda()

    # checkpoint_callback = ModelCheckpoint(
        # save_last=True,
        # save_top_k=1,
        # verbose=True,
        # monitor='val_acc',
        # mode='max'
    # )
    # early_stop_callback = EarlyStopping(
        # monitor='val_acc',
        # min_delta=0.00,
        # patience=500,
        # verbose=False,
        # mode='max')

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=200,
        verbose=False,
        mode='min'
    )

    wandb_logger = WandbLogger(
        name=f"{run_params.date} fold {run_params.fold}",
        project=run_params.project,
        entity=run_params.group,
        offline=True
    )
    trainer = pl.Trainer.from_argparse_args(
        run_params,
        callbacks=[checkpoint_callback, early_stop_callback],
        gpus=GPUS if DEVICE == 'cuda' else None,
        logger=wandb_logger
    )
    trainer.fit(model, datamodule=MyDataModule())

    trainer.test(datamodule=MyDataModule())
    trainer.validate(datamodule=MyDataModule())
    wandb.finish(0)
    wandb_logger
    if run_params.filename and wandb_logger.version:
        print(f'wandb id: {wandb_logger.version}')
        with open(run_params.filename, 'w') as f:
            f.write(f'{wandb_logger.version}\n{run_params.project}')


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--group", default='')
    parser.add_argument("--project", default='')
    parser.add_argument("--dataset", default='MUTAG')
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--folds", default=10, type=int)

    parser.add_argument("--nodes", default=6, type=int)
    parser.add_argument("--labels", default=7, type=int)
    parser.add_argument("--hidden", default=16, type=int)

    parser.add_argument("--filters", default=8, type=int)
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--hops", default=1, type=int)  # submask radius
    parser.add_argument("--kernel", default='wl', type=str)
    parser.add_argument("--k_type", default='v3.1', type=str)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--normalize", default=True, type=bool)
    parser.add_argument("--temp", default=False, type=bool)
    parser.add_argument("--auroc_test", default=False, type=bool)
    parser.add_argument("--auroc_val", default=False, type=bool)

    parser.add_argument("--pooling", default='add', type=str)

    parser.add_argument("--jsd_weight", default=1e4, type=float)
    parser.add_argument("--max_cc", default=True, type=bool)

    parser.add_argument("--max_epochs", default=2000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_graph", default=1e-2, type=float)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--avv", default=0, type=int)
    parser.add_argument("--mode", default='feature', type=str)
    parser.add_argument("--user", default='', type=str)
    parser.add_argument("--filename", default='', type=str)
    return parser


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[-2] == 'launch':
        params = json.loads(sys.argv[-1])
        default = get_arg_parser().parse_args([]).__dict__
        for k in params.keys():
            default[k] = params[k]
        params = Namespace(**default)
    else:
        params = get_arg_parser().parse_args()

    params.lr_graph = params.lr
    print(params)
    if params.gpu is not None:
        GPUS = params.gpu
        DEVICE = 'cuda'

    params.date = f'{datetime.utcnow():%Y-%m-%d %H:%MZ}'
    run_training_process_with_validation(params)
