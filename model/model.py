import math
import torch
import torch.nn as nn
from torch.nn import Parameter
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


class EfficientNet_b3(nn.Module):
    """
    batch size 64
    """
    def __init__(self, model_name='efficientnet-b3',num_classes=18):
        super(EfficientNet_b3, self).__init__()
        self.num_classes = num_classes
        self.model = EfficientNet.from_pretrained(model_name)
        print(self.model)
        n_features = self.model._fc.in_features
        self.model._fc = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)
        torch.nn.init.xavier_uniform_(self.model._fc.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model._fc.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        return self.model(inputs)
        
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

class PretrainModelTimmViTBasePath16(nn.Module):
    """
    batch size : 32
    timm pretrained model format
    https://fastai.github.io/timmdocs/
    """
    def __init__(self, model_name='vit_base_patch16_384', num_classes=18):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        print(self.model)

        n_features = self.model.head.in_features
        self.model.head = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.head.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.head.bias.data.uniform_(-stdv, stdv)

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

class PretrainModelTimmArc(nn.Module):
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
        self.model.classifier = torch.nn.Linear(in_features=n_features, out_features=n_features, bias=True)
        self.metric_fc = ArcMarginProduct(n_features, self.num_classes, s=30, m=0.5, easy_margin=False)

        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        #out = self.model.global_pool(x)
        #out = self.model(x)
        return self.model(x)


#####################################
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output