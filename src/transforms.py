import torch
import numpy as np


def min_max_transform(ts, min_data, max_data, range_needed):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts, n_dim=160, min_num=-128, max_num=127,
                 range_needed=(-1, 1)):
    ts_std = min_max_transform(ts,
                               min_data=min_num,
                               max_data=max_num,
                               range_needed=range_needed)
    sample_size = 800000
    bucket_size = int(sample_size / n_dim)
    new_ts = []
    for i in range(0, sample_size, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range,
                                       [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        relative_percentile = percentil_calc - mean
        new_ts.append(np.concatenate(
            [np.asarray([mean, std, std_top, std_bot, max_range]),
             percentil_calc, relative_percentile]))
    return np.asarray(new_ts)


class KaggleSignalTransform:
    def __init__(self, n_dim=160,
                 min_num=-128, max_num=127,
                 range_needed=(-1, 1)):
        self.n_dim = n_dim
        self.min_num = min_num
        self.max_num = max_num
        self.range_needed = range_needed

    def __call__(self, signal):
        signal = transform_ts(signal,
                              n_dim=self.n_dim,
                              min_num=self.min_num,
                              max_num=self.max_num,
                              range_needed=self.range_needed)
        return signal


class ToTensor:
    def __call__(self, x):
        x = torch.from_numpy(np.asarray(x, np.float32))
        return x
