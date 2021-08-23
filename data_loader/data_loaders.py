import os
from .datasets import MaskDataset
from torchvision import transforms
from base import BaseDataLoader
<<<<<<< HEAD
=======
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


>>>>>>> ce454becfabaa196c573028060fd4d16fe37cbd4

class MaskDataLoader(BaseDataLoader):
    """
    Mask data loader
    data_dir:str: data directory path "../input/data"
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, trsfm=False):
<<<<<<< HEAD
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')
        
=======
>>>>>>> ce454becfabaa196c573028060fd4d16fe37cbd4
        if not trsfm or not training:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                #[TODO] We should change normalize value by what pretrained model we use
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
<<<<<<< HEAD
        
        self.dataset = MaskDataset(csv_path=os.path.join(self.train_dir, 'train.csv'), transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataLoader = MaskDataLoader(data_dir="../input/data", batch_size=16)
    images, labels = next(iter(dataLoader))
    plt.figure(figsize=(12,12))
    for n, (image, label) in enumerate(zip(images, labels), start=1):
        plt.subplot(4,4,n)
        image=image.permute(1,2,0)
        plt.imshow(image)
        plt.title("{}".format(label))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
=======
        self.data_dir = os.path.abspath(data_dir) 
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = CustomDataset(csv_path='../input/data/train/trainV3.csv', images_folder=self.data_dir, transform=transform)
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
            try :
                png_path = path+".png"
                image = PIL.Image.open(os.path.join(self.images_folder, png_path))
            except :
                jpeg_path = path+".jpeg"
                image = PIL.Image.open(os.path.join(self.images_folder, jpeg_path))
        if self.transform is not None:
            image = self.transform(image)        
        return image, label

>>>>>>> ce454becfabaa196c573028060fd4d16fe37cbd4
