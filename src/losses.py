import torch
from torch import nn


def matthews_correlation_loss(y_pred, y_true):
    smooth = 1e-4

    y_pred_pos = torch.clamp(y_pred, 0.0, 1.0)
    y_pred_neg = 1 - y_pred_pos

    y_pos = torch.clamp(y_true, 0.0, 1.0)
    y_neg = 1 - y_pos

    tp = (y_pos * y_pred_pos).sum()
    tn = (y_neg * y_pred_neg).sum()

    fp = (y_neg * y_pred_pos).sum()
    fn = (y_pos * y_pred_neg).sum()

    numerator = (tp * tn - fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = numerator / (denominator + smooth)
    return torch.clamp(score, 0.0, 1.0)


class MatthewsCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return 1 - matthews_correlation_loss(output, target)


def fb_loss(preds, trues, beta):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0., 1.)


class FBLoss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


class MccBceLoss(nn.Module):
    def __init__(self, mcc_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.mcc_weight = mcc_weight
        self.bce_weight = bce_weight

        self.mcc_loss = FBLoss(1)
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        # print(target)
        if self.mcc_weight > 0:
            mcc = self.mcc_loss(output, target) * self.mcc_weight
        else:
            mcc = 0

        if self.bce_weight > 0:
            bce = self.bce_loss(output, target) * self.bce_weight
        else:
            bce = 0
        return mcc + bce
