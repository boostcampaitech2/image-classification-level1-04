import os
from albumentations.augmentations.functional import normalize
from torch.utils.data.sampler import Sampler
from torchvision.transforms.transforms import ToTensor
from .datasets import MaskSubmitDataset, DatasetForMask
from torchvision import transforms
from base import BaseDataLoader
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
from albumentations.pytorch.transforms import ToTensorV2
class AugMix(ImageOnlyTransform):
    """Augmentations mix to Improve Robustness and Uncertainty.
    Args:
        image (np.ndarray): Raw input image of shape (h, w, c)
        severity (int): Severity of underlying augmentation operators.
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly
          from [1, 3]
        alpha (float): Probability coefficient for Beta and Dirichlet distributions.
        augmentations (list of augmentations): Augmentations that need to mix and perform.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[A.HorizontalFlip()], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")

augs = [A.HorizontalFlip(always_apply=True),        
        A.Blur(always_apply=True),
        A.OneOf(
        [A.ShiftScaleRotate(always_apply=True),
        A.GaussNoise(always_apply=True)]
        ),        
        A.PiecewiseAffine(always_apply=True)]

transforms_train = A.Compose([
    A.CenterCrop(400, 260),    
    AugMix(width=3, depth=2, alpha=.2, p=1., augmentations=augs),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],  
                            max_pixel_value=255.0,
    p=1.0),
    ToTensorV2()
])
class DataLoaderForMask(BaseDataLoader):
    """
    Mask data loader
    data_dir:str: data directory path "../input/data"
    """
    def __init__(self, data_dir, batch_size, shuffle=False,
                validation_split=0.0, num_workers=1,
                training=True, trsfm=transforms_train, submit=False):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval') 
        self.validation_split = validation_split        
        if not trsfm or not training or submit:
                trsfm = A.Compose([
                ToTensorV2(),
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                A.CenterCrop(480, 260),                
            ])
        
        if not submit:
            self.shuffle = shuffle
            self.dataset = DatasetForMask(
                            csv_path=os.path.join(self.train_dir, 'train_for_Mask.csv'),
                            transform=trsfm)
        else:            
            self.shuffle = False
            self.dataset = MaskSubmitDataset()
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)    
        super().__init__(self.dataset, batch_size, self.shuffle, validation_split, num_workers)
    def _split_sampler(self, split):
        '''
        stratified train & val split
        '''
        if split == 0.0:
            #return None, None
            return ImbalancedDatasetSampler(self.dataset), None

        idx_full = np.arange(self.n_samples)
        labels = self.dataset.df[:, -1]

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert self.n_samples * split < self.n_samples, "validation set size is configured to be larger than entire dataset."
        
        train_idx, valid_idx = train_test_split(idx_full, stratify=labels, test_size=split, random_state=42)
        
        train_sampler = ImbalancedDatasetSampler(train_idx)
        valid_sampler = Sampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataLoader = DatasetForMask(data_dir="../input/data", batch_size=16)
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
