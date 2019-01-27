from argus import Model

from src.nn_modules import SimpleLSTM


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM
    }
