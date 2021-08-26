#%%
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from data_loader.data_loaders import DataLoaderForMask
import albumentations as A

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
        A.OneOf([A.ShiftScaleRotate(always_apply=True),
        A.GaussNoise(always_apply=True)]
        ),        
        A.PiecewiseAffine(always_apply=True)]

transforms_train = A.Compose([
    A.CenterCrop(400, 260),
    AugMix(width=3, depth=2, alpha=.2, p=1., augmentations=augs),    
    A.pytorch.transforms.ToTensorV2(),
])
trsfm = A.Compose([                
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                A.CenterCrop(400, 260),
                #A.HueSaturationValue(),
                A.HorizontalFlip(),                
                A.CLAHE(),                
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225],  
                                      max_pixel_value=255.0,
                p=1.0),
                ])        
if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    dataLoader = DataLoaderForMask(data_dir="../input/data", batch_size=16, trsfm=transforms_train)
    images, labels = next(iter(dataLoader))
    plt.figure(figsize=(12,12))
    for n, (image, label) in enumerate(zip(images["image"], labels), start=1):
        plt.subplot(4,4,n)        
        #image = cv2.Sobel(np.array(image),-1,0,1, ksize=3)
        #image = cv2.Scharr(np.array(image),-1,0,1)
        #image = cv2.Laplacian(np.array(image), -1)        
        image=image.permute(1, 2, 0)
        plt.imshow(image)
        plt.title("{}".format(label))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
#%%
