import albumentations as A
from albumentations.pytorch import ToTensorV2

# Change this value by what pretrained model you use
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]
# Mask Dataset's statics
# See EDA/2_EDA.ipynb
MEAN_Mask = [0.55800916, 0.51224077, 0.47767341]
STD_MASK = [0.21817792, 0.23804603, 0.25183411]


def transforms_select(method, MEAN=MEAN_IMAGENET, STD=STD_IMAGENET):
    lib =  {'DEFAULT' : A.Compose([
                            A.Resize(384, 384),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),

            'VIT_DEFAULT' : A.Compose([
                            A.Resize(384, 384),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),

            "TRNS001" : A.Compose([
                            A.Resize(512, 384),
                            A.HueSaturationValue(),
                            A.HorizontalFlip(p=1),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),

            "TRNS011" : A.Compose([
                            A.Resize(512//4, 224//4),
                            A.HueSaturationValue(),
                            A.HorizontalFlip(p=1),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),

            "VIT_TRNS001" : A.Compose([
                            A.Resize(384, 384),
                            A.HueSaturationValue(),
                            A.HorizontalFlip(p=1),
                            A.Normalize(mean=MEAN,
                                        std=STD),
                            ToTensorV2(),
                        ]),
            # Not Recommand using CLAHE
            "Cutout_Elastic_CLAHE_Transform" : A.Compose([
                                        A.Resize(384, 384),
                                        A.ElasticTransform(),
                                        A.Cutout(),
                                        A.CLAHE(),
                                        A.Normalize(mean=MEAN,
                                                    std=STD),
                                        ToTensorV2(),
                                    ]),

            "Cutout_Elastic_Transform" : A.Compose([
                                        A.Resize(384, 384),
                                        A.ElasticTransform(),
                                        A.Cutout(),
                                        A.Normalize(mean=MEAN,
                                                    std=STD),
                                        ToTensorV2(),
                                    ]),

            "TRNS004" : A.Compose([
                            A.Resize(384, 384), # FaceNet으로 찾은 평균적인 잘라낸 사진의 크기 (mean + 1.5 * std)
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