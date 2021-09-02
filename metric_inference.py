from pytorch_metric_learning.utils.inference import MatchFinder, InferenceModel
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from torchvision import datasets, transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np

labels_to_indices = c_f.get_labels_to_indices(dataset.targets)
model = torch.nn.DataParallel(resnet.resnet20())
checkpoint = torch.load("pytorch_resnet_cifar10/pretrained_models/resnet20-12fca82f.th")
model.load_state_dict(checkpoint['state_dict'])
model.module.linear = c_f.Identity() 
model.to(torch.device("cuda"))
print("done model loading")


match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
inference_model = InferenceModel(model, match_finder=match_finder)

# cars and frogs
classA, classB = labels_to_indices[1], labels_to_indices[6]

# create faiss index
inference_model.train_indexer(dataset)