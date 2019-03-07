from argus import Model

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM, Conv1dAttention


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM,
        'Conv1dAttention': Conv1dAttention
    }
