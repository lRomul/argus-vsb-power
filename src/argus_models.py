from argus import Model

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM, Conv1dLSTM


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM,
        'Conv1dLSTM': Conv1dLSTM
    }
