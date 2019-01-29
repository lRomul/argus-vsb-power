import os
import json
import random
import numpy as np
from pprint import pprint
import pyarrow.parquet as pq

from argus.callbacks import MonitorCheckpoint, \
    EarlyStopping, LoggingToFile, ReduceLROnPlateau

from torch.utils.data import DataLoader

from src.datasets import PowerDataset
from src.transforms import KaggleSignalTransform, ToTensor
from src.argus_models import PowerMetaModel


EXPERIMENT_NAME = 'random_search_simple_lstm_001'
VAL_FOLDS = [0]
TRAIN_FOLDS = [1, 2, 3, 4]
SAVE_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
START_FROM = 0


if __name__ == "__main__":
    for i in range(START_FROM, 10000):
        experiment_dir = f'{SAVE_DIR}/{i:04}'
        np.random.seed(i)
        random.seed(i)
        random_params = {
            'p_dropout': float(np.random.uniform(0.0, 1.0)),
            'seq_len': int(np.random.choice([160, 320, 640])),
            'batch_size': int(np.random.choice([32, 64, 128, 256])),
            'base_size': int(np.random.choice([16, 32, 64, 128])),
            'lr': float(np.random.choice([0.003, 0.001, 0.0003, 0.0001, 0.00003])),
            'patience': int(np.random.randint(10, 70)),
            'factor': float(np.random.uniform(0.3, 0.9)),
        }
        pprint(random_params)

        params = {
            'nn_module': ('SimpleLSTM', {
                'seq_len': random_params['seq_len'],
                'input_size': 19 * 3,
                'p_dropout': random_params['p_dropout'],
                'base_size': random_params['base_size']
            }),
            'loss': 'BCELoss',
            'optimizer': ('Adam', {'lr': random_params['lr']}),
            'device': 'cuda'
        }
        pprint(params)

        train_dataset = PowerDataset(TRAIN_FOLDS,
                                     preproc_signal_transform=KaggleSignalTransform(n_dim=random_params['seq_len']),
                                     signal_transform=ToTensor(),
                                     target_transform=ToTensor())
        val_dataset = PowerDataset(VAL_FOLDS,
                                   preproc_signal_transform=KaggleSignalTransform(n_dim=random_params['seq_len']),
                                   signal_transform=ToTensor(),
                                   target_transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=random_params['batch_size'], shuffle=True,
                                  drop_last=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=random_params['batch_size'], shuffle=False, num_workers=8)

        model = PowerMetaModel(params)

        callbacks = [
            MonitorCheckpoint(experiment_dir, monitor='val_mcc', max_saves=3, copy_last=False),
            EarlyStopping(monitor='val_mcc', patience=120),
            ReduceLROnPlateau(monitor='val_mcc', patience=random_params['patience'], factor=random_params['factor']),
            LoggingToFile(os.path.join(experiment_dir, 'log.txt')),
        ]

        with open(os.path.join(experiment_dir, 'random_params.json'), 'w') as outfile:
            json.dump(random_params, outfile)

        model.fit(train_loader,
                  val_loader=val_loader,
                  max_epochs=700,
                  callbacks=callbacks,
                  metrics=['mcc'])
