import os
import json
import pyarrow.parquet as pq

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import PowerDataset
from src.transforms import KaggleSignalTransform, ToTensor
from src.argus_models import PowerMetaModel
from src import config


EXPERIMENT_NAME = 'mcc_loss_lstm_001'
BATCH_SIZE = 64
SEQ_LEN = 320
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
FOLDS = config.FOLDS
PARAMS = {
    'nn_module': ('SimpleLSTM', {
        'seq_len': SEQ_LEN,
        'input_size': 19 * 3,
        'p_dropout': 0.3,
        'base_size': 64
    }),
    'loss': ('MccBceLoss', {'mcc_weight': 0.5, 'bce_weight': 0.5}),
    'optimizer': ('Adam', {'lr': 0.001}),
    'device': 'cuda'
}


def train_fold(save_dir, train_folds, val_folds):
    train_dataset = PowerDataset(train_folds,
                                 preproc_signal_transform=KaggleSignalTransform(n_dim=SEQ_LEN),
                                 signal_transform=ToTensor(),
                                 target_transform=ToTensor())
    val_dataset = PowerDataset(val_folds,
                               preproc_signal_transform=KaggleSignalTransform(n_dim=SEQ_LEN),
                               signal_transform=ToTensor(),
                               target_transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PowerMetaModel(PARAMS)

    callbacks = [
        MonitorCheckpoint(save_dir, monitor='val_mcc', max_saves=3, copy_last=False),
        EarlyStopping(monitor='val_mcc', patience=100),
        ReduceLROnPlateau(monitor='val_mcc', patience=30, factor=0.64, min_lr=1e-8),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    model.fit(train_loader,
              val_loader=val_loader,
              max_epochs=500,
              callbacks=callbacks,
              metrics=['mcc'])


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    else:
        print(f"Folder {SAVE_DIR} already exists.")

    with open(os.path.join(SAVE_DIR, 'source.py'), 'w') as outfile:
        outfile.write(open(__file__).read())

    with open(os.path.join(SAVE_DIR, 'params.json'), 'w') as outfile:
        json.dump(PARAMS, outfile)

    for i in range(len(FOLDS)):
        val_folds = [FOLDS[i]]
        train_folds = FOLDS[:i] + FOLDS[i + 1:]
        save_fold_dir = os.path.join(SAVE_DIR, f'fold_{FOLDS[i]}')
        print(f"Val folds: {val_folds}, Train folds: {train_folds}")
        print(f"Fold save dir {save_fold_dir}")
        train_fold(save_fold_dir, train_folds, val_folds)
