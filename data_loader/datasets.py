import os
import csv
import glob
import PIL
import pandas as pd
from torch.utils.data import Dataset
import torch


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
        self.trans_csv_path = os.path.join(self.dir_path, 'trainV4.csv')

        # if preprocessed trainV4.csv file doesnt' exists,
        # preprocess train.csv -> trainV4.csv
        if os.path.exists(self.trans_csv_path):
            pass
        else:
            self._makeCSV()

        self.df = pd.read_csv(self.trans_csv_path).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df[index]
        label = sample[-1]
        image = PIL.Image.open(sample[-2])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)

    def _makeCSV(self) :
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
                    if data["age"] > 60 :
                        label+=2
                    elif data["age"] >=30:
                        label+=1                
                    maskwriter.writerow([data["gender"], data["race"], data["age"], img_path, label])