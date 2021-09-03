from numpy.lib.function_base import average
import torch
from torchmetrics.functional import f1 as _f1

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).detach()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).detach()
    return correct / len(target)

def f1(output, target, num_classes=18):
    return _f1(output, target, num_classes=num_classes, average='macro')