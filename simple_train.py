#!/usr/bin/env python
# coding: utf-8

# In[1]:
import csv
import glob
import pandas as pd
import numpy as np
import PIL
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# # 1. 데이터 로드
# In[2]: 경로 설정
train_dir = '../input/data/train'
test_dir = '../input/data/eval'
save_dir = '../saved/models/'

# In[6]: 하이퍼파라미터
model_name = 'efficientnet_b3'
learning_rate = 1e-5
batch_size = 16
step_size = 5
epochs = 30
earlystop = 5

A_transform = {
    'train':
        A.Compose([
            A.Resize(384, 384),
            #A.RandomCrop(384, 384),
            A.HorizontalFlip(p=0.5),
            A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
            A.ElasticTransform(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    'valid':
        A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    'test':
        A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
}
# In[3]: 
class LoadCSV():
    def __init__(self, dir):
        self.dir = dir
        self.img_dir = train_dir + '/new_images/'
        self.origin_csv_path = train_dir + '/train.csv'
        self.trans_csv_path = train_dir + '/trans_train.csv'
        
        if not os.path.exists(self.trans_csv_path):
            self._makeCSV()
        self.df = pd.read_csv(self.trans_csv_path)
        #self.df = self.df[:200]
    def _makeCSV(self):
        with open(self.trans_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])

            df = pd.read_csv(self.origin_csv_path)
            for idx in range(len(df)):
                data = df.iloc[idx]
                img_path_base = os.path.join(os.path.join(self.img_dir, data['path']), '*')
                for img_path in glob.glob(img_path_base):
                    label = 0
                    if "incorrect" in img_path:
                        label+=6
                    elif 'normal' in img_path:
                        label+=12
                    elif data['gender']=='female':
                        label+=3
                    elif data['age'] >= 30 and data['age'] < 60:
                        label+=1
                    elif data['age'] >= 60:
                        label+=2
                    writer.writerow([img_path, label])
        f.close()

class MaskDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        class_id = torch.tensor(self.df['label'].iloc[idx])
        img = PIL.Image.open(self.df['path'].iloc[idx])
        img = np.array(img.convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, class_id


# # 2. 모델 설계
# In[4]:
class MyModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)

        n_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features=n_features, out_features=num_classes, bias=True)
        # n_features = self.model.head.in_features
        # self.model.head = torch.nn.Linear(in_features=n_features, out_features=self.num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.model.classifier.weight)
        stdv = 1/np.sqrt(self.num_classes)
        self.model.classifier.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        return self.model(x)


# In[7]: 데이터셋로드
mask_csv = LoadCSV(train_dir)
df_train, df_val = train_test_split(mask_csv.df, random_state=42, stratify=mask_csv.df.to_numpy()[:,-1])
mask_train = MaskDataset(df_train,  transform=A_transform['train'])
mask_valid = MaskDataset(df_val,  transform=A_transform['valid'])

train_loader = DataLoader(mask_train, batch_size=batch_size, drop_last=False, num_workers=8)
valid_loader = DataLoader(mask_valid, batch_size=batch_size, drop_last=False, num_workers=8)
dataloaders = {'train': train_loader, 'valid':valid_loader}


# In[8]:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel(model_name, 18).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)


# # 3. 학습
# In[9]:
today = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
if not os.path.exists(save_dir + today):
    os.makedirs(save_dir + today)


# In[10]:
earlystop_value = 0
best_model = copy.deepcopy(model.state_dict())
best_acc = 0
best_loss = 999999999

for epoch in range(epochs):
    if earlystop_value >= earlystop:
        break
    train_loss, valid_loss, train_acc_list, valid_acc_list = 0, 0, [],[]

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        with tqdm(dataloaders[phase], total=dataloaders[phase].__len__(), unit="batch") as train_bar:
            for inputs, labels in train_bar:
                train_bar.set_description(f"{phase} Epoch {epoch} ")
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (np.argmax(outputs, axis=1)== labels).mean()
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects / len(dataloaders[phase].dataset)
                train_bar.set_postfix(loss=epoch_loss, acc=epoch_acc)

        lr_scheduler.step()
        if phase=='valid':
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'{save_dir}{today}/baseline_{model_name}_lr{learning_rate}_stepLR{step_size}_batch{batch_size}_epoch{epoch}_valid_loss_{epoch_loss:.5f}.pt')
                earlystop_value = 0
            else:
                earlystop_value += 1
    
model.load_state_dict(best_model_wts)


# # 4. 추론
# In[11]:
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = PIL.Image.open(self.img_paths[index])
        image = np.array(image.convert("RGB"))
        if self.transform:
            image = self.transform(image=image)
            image = image['image']
        return image

    def __len__(self):
        return len(self.img_paths)


# In[12]: 제출코드 생성
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'new_images')

image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
dataset = TestDataset(image_paths, A_transform['test'])
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model.eval()
all_predictions = []
with tqdm(test_loader, total=test_loader.__len__(), unit="batch") as test_bar:
    for images in test_bar:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    
submission['ans'] = all_predictions

submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')


# In[ ]:

