import os
import csv
import glob
import PIL
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from .transforms import transforms_select


class BasicDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        assert transform is not None, "Set the transform on train set"
        self.transform = transform
        self.df = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = PIL.Image.open(self.df['path'].iloc[index])
        image = np.array(image.convert("RGB"))
        transformed = self.transform(image=image)
        image = transformed['image']
        return image, torch.tensor(self.df['label'].iloc[index])

    def get_labels(self):
        return self.df.iloc[:, -1]

class MaskSubmitDataset(Dataset):
    """
    Submission Dataset
    """
    def __init__(self, test_dir_path='/opt/ml/input/data/eval', crop=False, transform=None, image_glob=None):
        super().__init__()

        self.test_dir_path = test_dir_path
        if crop == False:
            self.image_dir_path = os.path.join(self.test_dir_path, 'images')
        else:
            self.image_dir_path = os.path.join(self.test_dir_path, 'images_face_crop')
     
        if image_glob:
            self.img_paths = [os.path.join(self.image_dir_path, img_id) for img_id in image_glob]
        else:
            self.df = pd.read_csv(os.path.join(self.test_dir_path, 'info.csv'))
            self.img_paths = [os.path.join(self.image_dir_path, img_id) for img_id in self.df['ImageID']]

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.img_paths[index])
        image = np.array(image.convert("RGB"))
        transformed = self.transform(image=image)
        image = transformed['image']
        return image

class MakeDataset():
    """
    dir_path: "../input/data/train"
    csv_path: original train.csv path
    """
    def __init__(self, csv_path, trsfm=None, defalut_trsfm=None):
        super().__init__()

        assert trsfm is not None, "Set the transform on train set"
        assert defalut_trsfm is not None, "Set the default transform"

        self.transform = trsfm
        self.defalut_transform = defalut_trsfm

        self.dir_path = os.path.dirname(csv_path)
        self.csv_path = csv_path
        self.img_dir_path = os.path.join(self.dir_path, 'images')
        self.trans_csv_path = os.path.join(self.dir_path, 'trans_train.csv') # origin : 'trans_train.csv'
        # See EDA/FixNote_Labeling_error.ipynb
        self.incorrect_labels = {'error_in_female' : ['006359', '006360', '006361', '006362', '006363', '006364'],
                                'error_in_male' : ['001498-1', '004432'],
                                'swap_normal_incorrect' : ['000020', '004418', '005227']}
        # if preprocessed trans_train.csv file doesnt' exists,
        # preprocess train.csv -> trans_train.csv
        if os.path.exists(self.trans_csv_path):
            pass
        else:
            self._makeCSV()

        self.df = pd.read_csv(self.trans_csv_path)
        self.class_weights = self._get_class_weight()

    def _makeCSV(self):
        df = pd.read_csv(self.csv_path)
        with open(self.trans_csv_path, 'wt', newline='') as csvfile:
            maskwriter = csv.writer(csvfile)
            maskwriter.writerow(["gender", "race", "age", "path", "label"])
            for index in range(len(df)) : 
                data = df.iloc[index]

                img_path_base = os.path.join(os.path.join(self.img_dir_path, data['path']), '*')
                for img_path in glob.glob(img_path_base):
                    # labeling
                    label = 0
                    if "incorrect" in img_path:
                        label+=6
                    elif "normal" in img_path:
                        label+=12
                    if data["gender"] =="female":
                        label+=3
                    if data["age"] >= 60 :
                        label+=2
                    elif data["age"] >=30 and data["age"] < 60:
                        label+=1
                    
                    # incorrect label fix
                    ## 1. female -> male
                    if data['id'] in self.incorrect_labels['error_in_female']:
                        label-=3
                    ## 2. male -> female
                    if data['id'] in self.incorrect_labels['error_in_male']:
                        label+=3
                    ## 3. mask <-> incorrect
                    if "incorrect" in img_path and data['id'] in self.incorrect_labels['swap_normal_incorrect']:
                        label+=6
                    if "normal" in img_path and data['id'] in self.incorrect_labels['swap_normal_incorrect']:
                        label-=6

                    maskwriter.writerow([data["gender"], data["race"], data["age"], img_path, label])

    def _get_class_weight(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                            classes=np.sort(np.unique(self.df.iloc[:, -1])), # label
                            y=self.df.iloc[:, -1])))
        return torch.tensor(list(class_weights.values()), dtype=torch.float)
    
    def split_validation(self, validation_split):
        df_train, df_val = train_test_split(self.df, test_size=validation_split, random_state=42, stratify=self.df.to_numpy()[:,-1])
        train = BasicDataset(df_train, transform=self.transform)
        val = BasicDataset(df_val, transform=self.defalut_transform)
        return train, val


class MakeAgeDataset():
    def __init__(self, csv_path, trsfm=None, defalut_trsfm=None):

        assert trsfm is not None, "Set the transform on train set"
        assert defalut_trsfm is not None, "Set the default transform"

        self.transform = trsfm
        self.defalut_transform = defalut_trsfm

        self.dir_path = os.path.dirname(csv_path)
        self.csv_path = csv_path
        self.img_dir_path = os.path.join(self.dir_path, 'images_face_crop')
        self.trans_csv_path = os.path.join(self.dir_path, 'trans_train_age.csv')
        self.incorrect_labels = {'error_in_female' : ['006359', '006360', '006361', '006362', '006363', '006364'],
                                'error_in_male' : ['001498-1', '004432'],
                                'swap_normal_incorrect' : ['000020', '004418', '005227']}

        if not os.path.exists(self.trans_csv_path):
            self._makeCSV()
        self.df = pd.read_csv(self.trans_csv_path)
        self.class_weights = self._get_class_weight()

    def _makeCSV(self):        
        with open(self.trans_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["gender", "age", "path", "label"])

            df = pd.read_csv(self.origin_csv_path)
            for idx in range(len(df)):
                data = df.iloc[idx]
                img_path_base = os.path.join(os.path.join(self.img_dir_path, data['path']), '*')
                if data['age'] <30:
                    continue
                for img_path in glob.glob(img_path_base):
                    label = 0
                    if data['gender']=='female':
                        label+=4
                    if data['age'] >= 30 and data['age'] < 40:
                        label+=0
                    elif data['age'] >= 40 and data['age'] < 50:
                        label+=1
                    elif data['age'] >= 50 and data['age'] < 60:
                        label+=2
                    elif data['age'] >= 60:
                        label+=3
                    # incorrect label fix
                    ## 1. female -> male
                    if data['id'] in self.incorrect_labels['error_in_female']:
                        label-=4
                    ## 2. male -> female
                    if data['id'] in self.incorrect_labels['error_in_male']:
                        label+=4
                    writer.writerow([data['gender'], data['age'], img_path, label])


    def _get_class_weight(self):
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                            classes=np.sort(np.unique(self.df.iloc[:, -1])), # label
                            y=self.df.iloc[:, -1])))
        return torch.tensor(list(class_weights.values()), dtype=torch.float)

    def split_validation(self, validation_split):
        df_train, df_val = train_test_split(self.df, test_size=validation_split, random_state=42, stratify=self.df.to_numpy()[:,-1])
        train = BasicDataset(df_train, transform=self.transform)
        val = BasicDataset(df_val, transform=self.defalut_transform)
        return train, val