import torch
from apex.fp16_utils import FP16_Optimizer

from argus import Model
from argus.utils import deep_to, deep_detach

from src.metrics import MatthewsCorrelation
from src.nn_modules import SimpleLSTM, Conv1dAvgPool, Conv1dLSTMAtt


class PowerMetaModel(Model):
    nn_module = {
        'SimpleLSTM': SimpleLSTM,
        'Conv1dAvgPool': Conv1dAvgPool,
        'Conv1dLSTMAtt': Conv1dLSTMAtt
    }

    def __init__(self, params):
        super().__init__(params)
        self.nn_module = self.nn_module.half()
        self.fp16_optimizer = FP16_Optimizer(self.optimizer,
                                             static_loss_scale=128.0)

    def prepare_batch(self, batch, device):
        input, target = batch
        input = deep_to(input, device,
                        dtype=torch.float16, non_blocking=True)
        target = deep_to(target, device,
                         dtype=torch.float16, non_blocking=True)
        return input, target

    def train_step(self, batch)-> dict:
        if not self.nn_module.training:
            self.nn_module.train()
        self.fp16_optimizer.zero_grad()
        input, target = self.prepare_batch(batch, self.device)
        prediction = self.nn_module(input)
        loss = self.loss(prediction, target)
        self.fp16_optimizer.backward(loss)
        self.fp16_optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        return {
            'prediction': self.prediction_transform(prediction),
            'target': target,
            'loss': loss.item()
        }
