import pyarrow.parquet as pq
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
    id_measurement_lst = []

    for id_measurement, group in metadata_df.groupby('id_measurement'):
        if id_measurement2fold[id_measurement] not in folds:
            continue

        targets = group['target'].tolist()
        signal_ids = group['signal_id'].tolist()

        signals_lst.append(signal_array[signal_ids])
        targets_lst.append(targets)
        id_measurement_lst.append(id_measurement)

    return signals_lst, targets_lst, id_measurement_lst


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
        signals_lst, self.targets_lst, self.id_measurement_lst = \
            get_samples(metadata_path, signal_path, folds_path, folds)

        with mp.Pool(N_WORKERS) as pool:
            self.signals_lst = pool.map(self.preproc_signal_transform, signals_lst)

    def __len__(self):
        return len(self.id_measurement_lst)

    def __getitem__(self, idx):
        signal = self.signals_lst[idx]
        target = self.targets_lst[idx]

        if self.transform is not None:
            signal, target = self.transform(signal, target)
        if self.signal_transform is not None:
            signal = self.signal_transform(signal)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target
