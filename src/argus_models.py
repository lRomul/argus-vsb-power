from argus import Model

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM
from src.losses import MatthewsCorrelationLoss, MccBceLoss


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM
    }
    loss = {
        'MatthewsCorrelationLoss': MatthewsCorrelationLoss,
        'MccBceLoss': MccBceLoss
    }
