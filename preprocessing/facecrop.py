import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os
from PIL import Image

# Eval
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
img_path = '/opt/ml/input/data/eval/images/'
new_img_dir = '/opt/ml/input/data/eval/images_facecrop/'

cnt = 0
os.makedirs(new_img_dir, exist_ok=True)

for paths in os.listdir(img_path):
    if paths[0] == '.': continue
    
    sub_dir = os.path.join(img_path, paths)
    print(sub_dir)
    
    img = Image.open(sub_dir)

    
    #mtcnn 적용
    boxes,probs = mtcnn.detect(img)
    
    # boxes 확인
    if len(probs) > 1: 
        print(boxes)
    if not isinstance(boxes, np.ndarray):
        print('Nope!')
        # 직접 crop
        #img=img[100:400, 50:350, :]
        img = img.crop((50,100,350,400))
        tmp = os.path.join(new_img_dir,paths)
        cnt += 1
        print(tmp)
        #plt.imsave(os.path.join(tmp, imgs), img)
        img.save(tmp)
    
    # boexes size 확인
    else:
        xmin = int(boxes[0, 0])-30
        ymin = int(boxes[0, 1])-30
        xmax = int(boxes[0, 2])+30
        ymax = int(boxes[0, 3])+30
        
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > 384: xmax = 384
        if ymax > 512: ymax = 512
        
        # img = img[ymin:ymax, xmin:xmax, :]
        '''
        _x _y
            x_, y_
        '''
        #img = img[_y:y_, _x:x_]
        #img.crop(_x, _y, x_, y_)
        img = img.crop((xmin,ymin,xmax,ymax))
        
        # sub_dir = os.path.join(img_path, paths)

        tmp = os.path.join(new_img_dir,paths)
        cnt += 1
        print(tmp)
        #plt.imsave(os.path.join(tmp, imgs), img)
        img.save(tmp)

# Train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
img_path = '/opt/ml/input/data/train/images'
new_img_dir = '/opt/ml/input/data/train/images_facecrop'

cnt = 0
os.makedirs(new_img_dir, exist_ok=True)

for paths in os.listdir(img_path):
    if paths[0] == '.': continue
    
    sub_dir = os.path.join(img_path, paths)
    
    for imgs in os.listdir(sub_dir):
        if imgs[0] == '.': continue
        
        img_dir = os.path.join(sub_dir, imgs)
        #img = cv2.imread(img_dir)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)LSAR2RGB
        img = Image.open(img_dir)

        
        #mtcnn 적용
        boxes,probs = mtcnn.detect(img)
        
        if not isinstance(boxes, np.ndarray):
            # 직접 crop
            #img=img[100:400, 50:350, :]
            img = img.crop((50,100,350,400))
        
        # boexes size 확인
        else:
            xmin = int(boxes[0, 0])-30
            ymin = int(boxes[0, 1])-30
            xmax = int(boxes[0, 2])+30
            ymax = int(boxes[0, 3])+30
            
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > 384: xmax = 384
            if ymax > 512: ymax = 512
            
            # img = img[ymin:ymax, xmin:xmax, :]
            '''
            _x _y
                   x_, y_
            '''
            #img = img[_y:y_, _x:x_]
            #img.crop(_x, _y, x_, y_)
            img = img.crop((xmin,ymin,xmax,ymax))
            
        if not os.path.exists(os.path.join(new_img_dir,paths)):
            os.makedirs(os.path.join(new_img_dir,paths))
        tmp = os.path.join(new_img_dir, paths)
        cnt += 1
        #plt.imsave(os.path.join(tmp, imgs), img)
        img.save(os.path.join(tmp, imgs))