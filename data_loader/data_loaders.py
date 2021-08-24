import os
from .datasets import MaskDataset, MaskSubmitDataset, DatasetForMask
from torchvision import transforms
from base import BaseDataLoader
import albumentations as A

class MaskDataLoader(BaseDataLoader):
    """
    Mask data loader
    data_dir:str: data directory path "../input/data"
    """
    def __init__(self, data_dir, batch_size, shuffle=True,
                validation_split=0.0, num_workers=1,
                training=True, trsfm=False, submit=False):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')        
        if not trsfm or not training or submit:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])
        
        if not submit:
            self.shuffle = shuffle
            self.dataset = MaskDataset(
                            csv_path=os.path.join(self.train_dir, 'train.csv'),
                            transform=trsfm)
        else:
            trsfm = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])
            self.shuffle = False
            self.dataset = MaskSubmitDataset(transform=trsfm)

        super().__init__(self.dataset, batch_size, self.shuffle, validation_split, num_workers)

class DataLoaderForMask(BaseDataLoader):
    """
    Mask data loader
    data_dir:str: data directory path "../input/data"
    """
    def __init__(self, data_dir, batch_size, shuffle=True,
                validation_split=0.0, num_workers=1,
                training=True, trsfm=False, submit=False):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')
        
        if not trsfm or not training or submit:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])
        
        if not submit:
            self.shuffle = shuffle
            self.dataset = DatasetForMask(
                            csv_path=os.path.join(self.train_dir, 'train_for_Mask.csv'),
                            transform=trsfm)
        else:
            trsfm = transforms.Compose([
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])
            self.shuffle = False
            self.dataset = MaskSubmitDataset(transform=trsfm)

        super().__init__(self.dataset, batch_size, self.shuffle, validation_split, num_workers)

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
