import albumentations as A
from albumentations.pytorch import ToTensorV2

# Change this value by what pretrained model you use
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


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
            # Write down any combination you want...
    }
    return lib[method]
    