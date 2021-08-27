import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import timm
from base import BaseModel
from efficientnet_pytorch import EfficientNet

class MaskModel(nn.Module):
    """
    Basic model format
    """
    def __init__(self, num_classes=18):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 93, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1) # flatten all dimensions except batch  # [B, 16, 125, 93]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PretrainModelTV(nn.Module):
    """
    torch vision pretrain model format
    https://pytorch.org/vision/stable/models.html
    """
    def __init__(self, model_name='resnet18',num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.model = getattr(torchvision.models, model_name)(pretrained=True)
        print("the number of class labels :", self.model.fc.weight.shape[0])
        self.model.fc = torch.nn.Linear(in_features=512,
                                            out_features=self.num_classes, bias=True)
        
        torch.nn.init.xavier_uniform_(self.model.conv1.weight)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)

class EfficientNet_b0(nn.Module):
    """
    batch size 64
    """
    def __init__(self, model_name='efficientnet-b0',num_classes=18):
        super(EfficientNet_b0, self).__init__()
        self.num_classes = num_classes
        self.model = EfficientNet.from_pretrained(model_name)

        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512 , 256),
            nn.Linear(256 , self.num_classes)
        )

    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x

class PretrainModelTimm(nn.Module):
    """
    batch size : 32
    timm pretrained model format
    https://fastai.github.io/timmdocs/
    """
    def __init__(self, model_name='efficientnet_b3', num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)
class PretrainModelTimmEcaNFNet12(nn.Module):
    """
    batch size : 32
    timm pretrained model format
    https://fastai.github.io/timmdocs/
    """
    def __init__(self, model_name="eca_nfnet_l2", num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.head.fc.in_features
        self.model.head.fc = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.head.fc.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.head.fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)

class PretrainModelTimmViT(nn.Module):
    def __init__(self, model_name="vit_large_r50_s32_384", num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)

        n_features = self.model.head.in_features
        self.model.head = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.head.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.head.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.model(x)