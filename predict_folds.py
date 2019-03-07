import os
from os.path import join
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp
import torch
import tqdm

from argus import load_model

from src.transforms import RawSignalScale, ToTensor, Crops, Compose
from src.argus_models import PowerMetaModel
from src.utils import make_dir
from src import config

EXPERIMENT_NAME = 'conv_att_001'
BATCH_SIZE = 32
SEQ_LEN = 524288
FOLDS_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
PREDICTION_DIR = f'/workdir/data/predictions/{EXPERIMENT_NAME}'
FOLDS = config.FOLDS
make_dir(PREDICTION_DIR)
THRESHOLD = 0.5
PRED_BATCH_SIZE = 512
DEVICE = 'cuda'

N_WORKERS = mp.cpu_count()

TRANSFORM = Compose([
    RawSignalScale(),
    ToTensor()
])
CROPS = Crops(SEQ_LEN)


def use_transforms(signal):
    signal = TRANSFORM(signal)
    signals = CROPS(signal)
    return signals


class Predictor:
    def __init__(self, model_paths, device='cpu'):
        self.models = [load_model(m, device=device) for m in model_paths]
        self.pool = mp.Pool(N_WORKERS)

    def __call__(self, signals):
        tensors = self.pool.map(use_transforms, signals)

        probs = []
        for tensor in tensors:
            tensor = torch.stack(tensor, dim=0)
            tensor = tensor.to(self.models[0].device)
            with torch.no_grad():
                prob = self.models[0].nn_module(tensor)
                for model in self.models[1:]:
                    prob += model.nn_module(tensor)
                prob = prob.cpu().numpy() / len(self.models)
                prob = prob.mean(axis=0)
                probs += prob.tolist()

        return probs

    def __del__(self):
        self.pool.close()


def pred_val_fold(model_paths, fold, train_signal):
    metadata_df = pd.read_csv(config.METADATA_TRAIN_PATH)
    folds_df = pd.read_csv(config.TRAIN_FOLDS_PATH)

    predictor = Predictor(model_paths, device=DEVICE)

    id_measurement2fold = dict()
    for _, row in folds_df.iterrows():
        id_measurement2fold[row.id_measurement] = row.fold

    signal_id_lst = []
    probs_lst = []

    signals = []
    for id_measurement, group in tqdm.tqdm(metadata_df.groupby('id_measurement')):
        if id_measurement2fold[id_measurement] not in [fold]:
            continue

        signal_ids = group['signal_id'].tolist()

        signal_id_lst += signal_ids
        signals.append(train_signal[signal_ids].copy())

        if len(signals) == PRED_BATCH_SIZE:
            probs = predictor(signals)
            probs_lst += probs
            signals = []

    if signals:
        probs = predictor(signals)
        probs_lst += probs

    probs_df = pd.DataFrame({'signal_id': signal_id_lst, 'target': probs_lst})
    fold_prediction_dir = join(PREDICTION_DIR, f'fold_{fold}', 'val')
    make_dir(fold_prediction_dir)
    probs_df.to_csv(join(fold_prediction_dir, 'probs.csv'), index=False)


def pred_test_fold(model_paths, fold, test_signal):
    metadata_df = pd.read_csv(config.METADATA_TEST_PATH)
    predictor = Predictor(model_paths, device=DEVICE)

    signal_id_lst = []
    probs_lst = []

    signals = []
    for id_measurement, group in tqdm.tqdm(metadata_df.groupby('id_measurement')):
        signal_ids = group['signal_id'].tolist()

        signal_id_lst += signal_ids
        signals.append(test_signal[[si - 8712 for si in signal_ids]].copy())

        if len(signals) == PRED_BATCH_SIZE:
            probs = predictor(signals)
            probs_lst += probs
            signals = []

    if signals:
        probs = predictor(signals)
        probs_lst += probs

    probs_df = pd.DataFrame({'signal_id': signal_id_lst, 'target': probs_lst})
    fold_prediction_dir = join(PREDICTION_DIR, f'fold_{fold}', 'test')
    make_dir(fold_prediction_dir)
    probs_df.to_csv(join(fold_prediction_dir, 'probs.csv'), index=False)


def get_model_paths(dir_path, window_size=5):
    model_scores = []
    for model_name in os.listdir(dir_path):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', model_name)
        if score is not None:
            score = score.group(0)[1:-4]
            model_scores.append((model_name, score))
    model_scores = sorted(model_scores, key=lambda x: x[0])

    max_score = 0
    max_score_index = 0
    for i in range(len(model_scores) - window_size):
        scores = [float(s[1]) for s in model_scores[i: i + window_size]]
        score = np.mean(scores)
        if score >= max_score:
            max_score = score
            max_score_index = i

    model_paths = [s[0] for s in
                   model_scores[max_score_index: max_score_index + window_size]]
    return [join(dir_path, p) for p in model_paths]


if __name__ == "__main__":
    train_signal = pq.read_pandas(config.TRAIN_PARQUET_PATH)
    train_signal = train_signal.to_pandas().values.T

    test_signal = pq.read_pandas(config.TEST_PARQUET_PATH)
    test_signal = test_signal.to_pandas().values.T

    for i in range(len(FOLDS)):
        print("Predict fold", FOLDS[i])
        fold_dir = os.path.join(FOLDS_DIR, f'fold_{FOLDS[i]}')
        model_paths = get_model_paths(fold_dir)
        print("Model paths", model_paths)
        print("Val predict")
        pred_val_fold(model_paths, FOLDS[i], train_signal)
        print("Test predict")
        pred_test_fold(model_paths, FOLDS[i], test_signal)
