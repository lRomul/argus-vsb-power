import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config

RANDOM_SEED = 42


def make_train_folds(metadata_train_path, n_folds):
    meta_train_df = pd.read_csv(metadata_train_path)

    targets_lst = []
    id_measurement_lst = []

    for id_measurement, group in meta_train_df.groupby('id_measurement'):
        targets = group['target'].tolist()
        targets_lst.append(targets)
        id_measurement_lst.append(id_measurement)

    skf = StratifiedKFold(n_splits=n_folds,
                          shuffle=True,
                          random_state=RANDOM_SEED)
    targets = [int(any(t)) for t in targets_lst]

    meas_id_lst = []
    fold_lst = []
    for fold, (_, fold_index) in enumerate(skf.split(targets, targets)):
        for index in fold_index:
            meas_id_lst.append(id_measurement_lst[index])
            fold_lst.append(fold)

    train_folds_df = pd.DataFrame({'id_measurement': meas_id_lst,
                                   'fold': fold_lst})
    train_folds_df.sort_values('id_measurement', inplace=True)
    return train_folds_df


if __name__ == '__main__':
    random.seed(RANDOM_SEED)

    train_folds_df = make_train_folds(config.METADATA_TRAIN_PATH,
                                      config.N_FOLDS)
    train_folds_df.to_csv(config.TRAIN_FOLDS_PATH, index=False)
    print(f"Folds saved to '{config.TRAIN_FOLDS_PATH}'")
