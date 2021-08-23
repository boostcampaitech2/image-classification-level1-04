from torchvision import datasets, transforms
from base import BaseDataLoader
import os
import PIL
import pandas as pd
from torch.utils.data import Dataset
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CIFAR10DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, trsfm=False):
        if not trsfm or not training:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class MaskDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, trsfm=False):
        if not trsfm or not training:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.data_dir = data_dir
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CustomDataset(csv_path='../../input/data/train/trainV3.csv', images_folder='../../input/data/train/images', transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CustomDataset(Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):        
        path = self.df.iloc[index]["path"]   
        label=self.df.iloc[index]["label"]        
        try :
            jpg_path = path+".jpg"
            image = PIL.Image.open(os.path.join(self.images_folder, jpg_path))
        except:
            png_path = path+".png"
            image = PIL.Image.open(os.path.join(self.images_folder, png_path))
        if self.transform is not None:
            image = self.transform(image)        
        return image, label

