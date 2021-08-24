import os
import csv
import glob
import PIL
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.utils import class_weight

class MaskDataset(Dataset):
    """
    dir_path: "../input/data/train"
    csv_path: original train.csv path
    """
    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.transform = transform
        self.dir_path = os.path.dirname(csv_path)
        self.csv_path = csv_path
        self.img_dir_path = os.path.join(self.dir_path, 'images')
        self.trans_csv_path = os.path.join(self.dir_path, 'trans_train.csv')

        # if preprocessed trainV4.csv file doesnt' exists,
        # preprocess train.csv -> trainV4.csv
        if os.path.exists(self.trans_csv_path):
            pass
        else:
            self._makeCSV()

        self.df = pd.read_csv(self.trans_csv_path).values
        self.class_weights = self._get_class_weight()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df[index]
        label = sample[-1]
        image = PIL.Image.open(sample[-2])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)

    def _makeCSV(self):
        df = pd.read_csv(self.csv_path)
        with open(self.trans_csv_path, 'wt', newline='') as csvfile:
            maskwriter = csv.writer(csvfile)
            maskwriter.writerow(["gender", "race", "age", "path", "label"])
            for index in range(len(df)) : 
                data = df.iloc[index]

                img_path_base = os.path.join(os.path.join(self.img_dir_path, data['path']), '*')
                for img_path in glob.glob(img_path_base):
                    label = 0
                    if "incorrect" in img_path :
                        label+=6
                    elif "normal" in img_path :
                        label+=12
                    if data["gender"] =="female":
                        label+=3
                    if data["age"] >= 60 :
                        label+=2
                    elif data["age"] >=30 and data["age"] < 60:
                        label+=1                
                    maskwriter.writerow([data["gender"], data["race"], data["age"], img_path, label])

    def _get_class_weight(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                            classes=np.sort(np.unique(self.df[:, -1])), # label
                            y=self.df[:, -1])))
        return torch.tensor(list(class_weights.values()), dtype=torch.float)

class MaskSubmitDataset(Dataset):
    """
    Submission Dataset
    """
    def __init__(self, test_dir_path='/opt/ml/input/data/eval', transform=None):
        super().__init__()

        self.test_dir_path = test_dir_path
        self.image_dir_path = os.path.join(self.test_dir_path, 'images')
        self.df = pd.read_csv(os.path.join(self.test_dir_path, 'info.csv'))
        self.img_paths = [os.path.join(self.image_dir_path, img_id) \
                            for img_id in self.df.ImageID]

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

class DatasetForMask(Dataset):
    """
    dir_path: "../input/data/train"
    csv_path: original train.csv path
    """
    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.transform = transform
        self.dir_path = os.path.dirname(csv_path)
        self.csv_path = csv_path
        self.img_dir_path = os.path.join(self.dir_path, 'images')
        self.trans_csv_path = os.path.join(self.dir_path, 'train_for_Mask.csv')

        # if preprocessed trainV4.csv file doesnt' exists,
        # preprocess train.csv -> trainV4.csv
        if os.path.exists(self.trans_csv_path):
            pass
        else:
            self._makeCSV()

        self.df = pd.read_csv(self.trans_csv_path).values
        self.class_weights = self._get_class_weight()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df[index]
        label = sample[-1]
        img_extension= ['jpg', 'jpeg', 'png']
        no_extension= os.path.join(self.img_dir_path, sample[-2])        
        for extension in img_extension :
            if os.path.exists(no_extension+'.'+extension) :        
                image_path = no_extension+'.'+extension                
                break        
        image = PIL.Image.open(image_path) 
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)  

    def _get_class_weight(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                            classes=np.sort(np.unique(self.df[:, -1])), # label
                            y=self.df[:, -1])))
        return torch.tensor(list(class_weights.values()), dtype=torch.float)