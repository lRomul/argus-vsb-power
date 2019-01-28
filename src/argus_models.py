from argus import Model

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM
    }
