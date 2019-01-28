import numpy as np
from sklearn.metrics import confusion_matrix

from argus.metrics.metric import Metric


class WrongShapesException(Exception):
    pass


def check_shapes(y_true, y_pred):
    '''
    Check array shapes
    Parameters
    ----------
    y_true: numpy.ndarray
        Target array
    y_pred: numpy.ndarray
        Predicion array
    '''

    if not y_true.shape == y_pred.shape:
        raise WrongShapesException("Array shapes are inconsistent: "
                                   "y_true: {} "
                                   "y_pred: {}".format(y_true.shape, y_pred.shape))


def align_shape(*arrays):
    '''
    Make all the arrays 2-dimensional
    Parameters
    ----------
    arrays: list of numpy.ndarray
        Multiple arrays
    Returns
    -------
    reshaped_arrays: list of numpy.ndarray
        Arrays reshaped to (-1, 1)
    '''
    return [arr.reshape(-1, 1) for arr in arrays if arr.ndim == 1]


def confusion_binary(y_true, y_pred):

    confmat = confusion_matrix(y_true, y_pred)

    confmat = confmat.astype(float)

    # TODO: check confmat
    true_negative = confmat[0, 0]
    false_negative = confmat[1, 0]

    true_positive = confmat[1, 1]
    false_positive = confmat[0, 1]

    return true_positive, true_negative, false_positive, false_negative


def mcc(y_true, y_pred):
    '''
    Matthews Correlation Coefficient
    Parameters
     ----------
     y_true: numpy.ndarray
        Targets
    y_pred: numpy.ndarray
        Class predictions (0 or 1 values only)
    Returns
    ------
    score: float
        Matthews Correlation Coefficient score
    References
    ----------
    .. [1] https://lettier.github.io/posts/2016-08-05-matthews-correlation-coefficient.html
    .. [2] https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    '''

    # Check shapes
    check_shapes(y_true, y_pred)
    y_true, y_pred = align_shape(y_true, y_pred)

    # Confusion matrix values
    tp, tn, fp, fn = confusion_binary(y_true, y_pred)

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))

    return numerator / denominator


class MatthewsCorrelation(Metric):
    name = 'mcc'
    better = 'max'

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        trg = step_output['target'].cpu().numpy()

        self.predictions.append(pred.ravel())
        self.targets.append(trg.ravel())

    def compute(self):
        y_true = np.concatenate(self.targets, axis=0)
        y_pred = np.concatenate(self.predictions, axis=0)
        y_true = (y_true > 0.5).astype(int)
        y_pred = (y_pred > 0.5).astype(int)
        score = mcc(y_true, y_pred)
        return score if not np.isnan(score) else 0
