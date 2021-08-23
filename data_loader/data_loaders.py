import os
from .datasets import MaskDataset
from torchvision import transforms
from base import BaseDataLoader

class MaskDataLoader(BaseDataLoader):
    """
    Mask data loader
    data_dir:str: data directory path "../input/data"
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, trsfm=False):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')
        
        if not trsfm or not training:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                #[TODO] We should change normalize value by what pretrained model we use
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
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
