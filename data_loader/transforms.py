import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Change this value by what pretrained model you use
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
MEAN_MASK = [0.55800916, 0.51224077, 0.47767341]
STD_MASK = [0.21817792, 0.23804603, 0.25183411]

def transforms_select(method, MEAN=MEAN_IMAGENET, STD=STD_IMAGENET):
    lib =  {'DEFAULT' : A.Compose([
                            A.Resize(384, 384),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),
                        
            "TRNS001" : A.Compose([
                            A.Resize(384, 384),
                            A.HueSaturationValue(),
                            A.HorizontalFlip(p=1),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),
            "TRNS002" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.OneOf([
                                A.MotionBlur(p=1),
                                A.OpticalDistortion(p=1),
                                A.GaussNoise(p=1)                 
                            ], p=1),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),
            "TRNS003" : A.Compose([
                            A.Resize(512, 512),
                            A.Resize(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),

            "TRNS004" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
                            A.ElasticTransform(),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),
            "TRNS005" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
                            A.ElasticTransform(),
                            A.Normalize(mean=MEAN_MASK,
                                        std=STD_MASK),
                            ToTensorV2(),
                    ]),
            "TRNS006" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            
                            A.ElasticTransform(),
                            A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                            A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),

            "TRNS007" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.ColorJitter(),
                            A.Cutout(num_holes=8, max_h_size=32,max_w_size=32),
                            A.ElasticTransform(),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),
                
            "TRNS008" : A.Compose([
                            A.Resize(512, 512),
                            A.RandomCrop(384, 384),
                            A.HorizontalFlip(p=0.5),
                            A.Cutout(num_holes=16, max_h_size=32,max_w_size=32),
                            A.ElasticTransform(),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                    ]),
            # Write down any combination you want...
    }
    return lib[method]
