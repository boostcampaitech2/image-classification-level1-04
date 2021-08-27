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
                        A.Normalize(mean=MEAN,
                                    std=STD),
                        ToTensorV2(),
                    ]),
            # Write down any combination you want...
    }
    return lib[method]
