#%%
import torch
import pandas as pd 
import PIL
from PIL import Image
import os 
import csv
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
def makeCSV(csv_path) :
    df = pd.read_csv(csv_path)
    with open('trainV3.csv', 'wt', newline='') as csvfile:
        maskwriter = csv.writer(csvfile)        
        images_name = ["mask1", "mask2", "mask3", "mask4", "mask5" , "normal", "incorrect_mask"]
        maskwriter.writerow(["gender", "race", "age", "path", "label"], )
        for index in range(len(df)) : 
            data = df.iloc[index]                
            for image_name in images_name :  
                label = 0
                if "incorrect" in image_name :
                    label+=6
                elif "normal" in image_name :
                    label+=12
                if data["gender"] =="female":
                    label+=3
                if data["age"] > 60 :
                    label+=2
                elif data["age"] >=30:
                    label+=1
                maskwriter.writerow([data["gender"], data["race"], data["age"], data["path"]+"/"+image_name, label])
makeCSV('../../input/data/train/train.csv')

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
            png_path = path+".png"
            image = PIL.Image.open(os.path.join(self.images_folder, png_path))
        if self.transform is not None:
            image = self.transform(image)        
        return image, label
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])
data=CustomDataset(csv_path='../../input/data/train/trainV3.csv', images_folder='../../input/data/train/images', transform=transform)

dataLoader = DataLoader(dataset=data, batch_size=16, shuffle=True)
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
# %%
