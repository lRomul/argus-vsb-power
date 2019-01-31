import os
from os.path import join
import re
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp
import torch

from argus import load_model

from src.transforms import KaggleSignalTransform, ToTensor
from src.argus_models import PowerMetaModel
from src.utils import make_dir
from src import config


EXPERIMENT_NAME = 'simple_lstm_001'
PRED_BATCH_SIZE = 512
SEQ_LEN = 320
FOLDS_DIR = f'/workdir/data/experiments/{EXPERIMENT_NAME}'
PREDICTION_DIR = f'/workdir/data/predictions/{EXPERIMENT_NAME}'
FOLDS = config.FOLDS
make_dir(PREDICTION_DIR)
THRESHOLD = 0.5

N_WORKERS = mp.cpu_count()


TRANSFORM = KaggleSignalTransform(n_dim=SEQ_LEN)
TO_TENSOR = ToTensor()


def use_transforms(signal):
    signal = TRANSFORM(signal)
    tensor = TO_TENSOR(signal)
    return tensor


class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model.nn_module.eval()

    def __call__(self, signals):
        with mp.Pool(N_WORKERS) as pool:
            tensors = pool.map(use_transforms, signals)

        tensor = torch.stack(tensors, dim=0)
        tensor = tensor.to(self.model.device)

        with torch.no_grad():
            probs = self.model.nn_module(tensor)
            probs = probs.cpu().numpy().flatten()
            return probs


def pred_val_fold(model_path, fold, train_signal):
    metadata_df = pd.read_csv(config.METADATA_TRAIN_PATH)
    folds_df = pd.read_csv(config.TRAIN_FOLDS_PATH)

    predictor = Predictor(model_path)

    id_measurement2fold = dict()
    for _, row in folds_df.iterrows():
        id_measurement2fold[row.id_measurement] = row.fold

    signal_id_lst = []
    probs_lst = []

    signals = []
    for id_measurement, group in metadata_df.groupby('id_measurement'):
        if id_measurement2fold[id_measurement] not in [fold]:
            continue

        signal_ids = group['signal_id'].tolist()

        signal_id_lst += signal_ids
        signals.append(train_signal[signal_ids].copy())

        if len(signals) == PRED_BATCH_SIZE:
            probs = predictor(signals).tolist()
            probs_lst += probs
            signals = []

    if signals:
        probs = predictor(signals).tolist()
        probs_lst += probs

    probs_df = pd.DataFrame({'signal_id': signal_id_lst, 'target': probs_lst})
    fold_prediction_dir = join(PREDICTION_DIR, f'fold_{fold}', 'val')
    make_dir(fold_prediction_dir)
    probs_df.to_csv(join(fold_prediction_dir, 'probs.csv'), index=False)


def get_best_model_path(dir_path):
    model_scores = []
    for model_name in os.listdir(dir_path):
        score = re.search(r'-(\d+(?:\.\d+)?).pth', model_name)
        if score is not None:
            score = score.group(0)[1:-4]
            model_scores.append((model_name, score))
    model_score = sorted(model_scores, key=lambda x: x[1])
    best_model_name = model_score[-1][0]
    best_model_path = os.path.join(dir_path, best_model_name)
    return best_model_path


if __name__ == "__main__":
    train_signal = pq.read_pandas(config.TRAIN_PARQUET_PATH)
    train_signal = train_signal.to_pandas().values.T

    test_signal = pq.read_pandas(config.TEST_PARQUET_PATH)
    test_signal = test_signal.to_pandas().values.T

    for i in range(len(FOLDS)):
        print("Predict fold", FOLDS[i])
        fold_dir = os.path.join(FOLDS_DIR, f'fold_{FOLDS[i]}')
        best_model_path = get_best_model_path(fold_dir)
        print("Model path", best_model_path)
        print("Val predict")
        pred_val_fold(best_model_path, FOLDS[i], train_signal)
