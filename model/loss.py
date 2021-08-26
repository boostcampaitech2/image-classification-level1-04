import torch.nn.functional as F
from torchmetrics.functional import f1 as _f1


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target, weight=None):
    return F.cross_entropy(output, target, weight=weight)

def f1(output, target, num_classes=18):
    return _f1(output, target, num_classes=num_classes, average='macro')