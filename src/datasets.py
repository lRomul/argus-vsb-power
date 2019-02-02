import pyarrow.parquet as pq
import torch
import pandas as pd
import multiprocessing as mp
from torch.utils.data import Dataset

from src import config


N_WORKERS = mp.cpu_count()


def get_samples(metadata_path, signal_path, folds_path, folds):
    metadata_df = pd.read_csv(metadata_path)
    signal_array = pq.read_pandas(signal_path)
    signal_array = signal_array.to_pandas().values.T
    folds_df = pd.read_csv(folds_path)

    id_measurement2fold = dict()
    for i, row in folds_df.iterrows():
        id_measurement2fold[row.id_measurement] = row.fold

    signals_lst = []
    targets_lst = []

    for _, row in metadata_df.iterrows():
        if id_measurement2fold[row.id_measurement] not in folds:
            continue

        signals_lst.append(signal_array[row.signal_id])
        targets_lst.append(row.target)

    return signals_lst, targets_lst


class PowerDataset(Dataset):
    def __init__(self, folds,
                 metadata_path=config.METADATA_TRAIN_PATH,
                 signal_path=config.TRAIN_PARQUET_PATH,
                 folds_path=config.TRAIN_FOLDS_PATH,
                 transform=None,
                 preproc_signal_transform=None,
                 signal_transform=None,
                 target_transform=None):
        super().__init__()
        self.folds = folds
        self.transform = transform
        self.preproc_signal_transform = preproc_signal_transform
        self.signal_transform = signal_transform
        self.target_transform = target_transform
        self.signals_lst, self.targets_lst = \
            get_samples(metadata_path, signal_path, folds_path, folds)

        if self.preproc_signal_transform is not None:
            with mp.Pool(N_WORKERS) as pool:
                self.signals_lst = pool.map(self.preproc_signal_transform, self.signals_lst)

    def __len__(self):
        return len(self.signals_lst)

    def __getitem__(self, idx):
        signal = self.signals_lst[idx].copy()
        target = [self.targets_lst[idx], ]

        if self.transform is not None:
            signal, target = self.transform(signal, target)
        if self.signal_transform is not None:
            signal = self.signal_transform(signal)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target
