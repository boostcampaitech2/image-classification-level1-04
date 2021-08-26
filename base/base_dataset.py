import os
import csv
import glob
import PIL
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.utils import class_weight
import cv2
from torchsampler import ImbalancedDatasetSampler
class BaseDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        super().__init__()
        self.transform = transform
        self.dir_path = os.path.dirname(csv_path)
        self.csv_path = csv_path
        self.img_dir_path = os.path.join(self.dir_path, 'images')
        self.trains_csv_path = os.path.join(self.dir_path, 'train_for_Mask.csv')

        # if preprocessed trainV4.csv file doesnt' exists,
        # preprocess train.csv -> trainV4.csv
        if os.path.exists(self.trains_csv_path):
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
        image = cv2.imread(image_path)
        #image = PIL.Image.open(image_path) 
        if self.transform is not None:
            image = self.transform(image=image)
        return image, torch.tensor(label)  

    def _get_class_weight(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                            classes=np.sort(np.unique(self.df[:, -1])), # label
                            y=self.df[:, -1])))
        return torch.tensor(list(class_weights.values()), dtype=torch.float)