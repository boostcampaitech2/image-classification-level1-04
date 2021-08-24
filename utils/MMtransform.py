import albumentations as A
import torch


def cutmix_for_mask(data, target, old_data, old_target):
    indices = torch.randperm(data.size(0))
    shuffled_old_target = old_target[indices]
    size = data.size()
    W = size[2]

    new_data = data.clone()
    new_data[:, :, :, : W // 2] = old_data[indices, :, :, W // 2 :]

    Iam = 0.5
    targets = (target, shuffled_old_target, Iam)

    return new_data, targets
