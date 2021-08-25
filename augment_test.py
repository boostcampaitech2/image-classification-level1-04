#%%
import cv2
import numpy as np
from data_loader.data_loaders import DataLoaderForMask
import albumentations as A
trsfm = A.Compose([                
                # Modify this value by what pretrained model you use
                # ref: https://pytorch.org/vision/stable/models.html
                A.CenterCrop(400, 260),
                A.HorizontalFlip(),
                A.CLAHE(),                
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                ])        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataLoader = DataLoaderForMask(data_dir="../input/data", batch_size=16, trsfm=trsfm)
    images, labels = next(iter(dataLoader))
    plt.figure(figsize=(12,12))
    for n, (image, label) in enumerate(zip(images["image"], labels), start=1):
        plt.subplot(4,4,n)        
        #image = cv2.Sobel(np.array(image),-1,0,1, ksize=3)
        #image = cv2.Scharr(np.array(image),-1,0,1)
        image = cv2.Laplacian(np.array(image), -1)
        plt.imshow(image)
        plt.title("{}".format(label))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
#%%
