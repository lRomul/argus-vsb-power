from argus import Model

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM, Conv1dAvgPool, Conv1dLSTMAtt


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM,
        'Conv1dAvgPool': Conv1dAvgPool,
        'Conv1dLSTMAtt': Conv1dLSTMAtt
    }
