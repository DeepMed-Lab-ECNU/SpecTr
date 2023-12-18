from sklearn.metrics import recall_score,f1_score
import numpy as np
from skimage.metrics import hausdorff_distance
from sklearn.metrics import jaccard_score
from einops import rearrange

def iou(y_hat, y):
    y_hat = y_hat.reshape(-1)
    y =y.reshape(-1)
    return jaccard_score(y_hat, y)

def eval_f1score(pred, label):    
    final_score = f1_score(label, pred, average='macro')
    return final_score


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:
    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, eps=1e-7, beta=1, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty and reference_empty:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

    return float(((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn + eps))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, eps=1e-7, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp + eps / (tp + fp + fn + eps))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, eps=1e-7, **kwargs):
    """TP / (TP + FN)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(tp + eps/ (tp + fn + eps))


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, eps=1e-7, **kwargs):
    """TN / (TN + FP)"""
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(tn + eps/ (tn + fp + eps))


# def f1_score(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1, eps=1e-7, **kwargs):
#     """TN / (TN + FP)"""
#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)
#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#
#     return ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)


def auc(test=None, reference=None):
    return roc_auc_score(reference, test)


# ((1 + beta ** 2) * tp + eps) \ ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)


def hausdorff_distance_case(pred, label):
    pred,label = pred.astype(np.uint8),label.astype(np.uint8)
    assert pred.shape[0] == 1 ,"one class be predicted"
    
    return hausdorff_distance(pred.reshape(-1),label.reshape(-1))
    
if __name__ == '__main__':
    pred = [0, 1, 2, 0, 1, 2]
    label = [2, 1, 0, 2, 1, 0]
    score = recall_score(pred, label, average='macro')
    print(score)