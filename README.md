# Boostcamp AI-Tech 2nd - Master Slave Team

Level 1 P-Stage Image Classification Project Repository

### Members
- 김대웅(팀장) : 스케쥴 및 할 일 관리, 전체 프레임워크 잡기
- 나요한 : Augmentation 실험, 기본 모델들의 성능 평가 및 모델 구현에 기여
- 박준수 : Augmentation 실험, 기본 모델들의 성능 평가 및 모델 구현에 기여
- 이호영 : 데이터 EDA, 추가 데이터셋 확보에 기여
- 최한준 : 데이터 EDA, 추가 데이터셋 확보에 기여
- 추창한 : 멀티 모델 실험, 모델의 성능 저하원인 분석
- 한진 : 멀티 모델 실험, 모델의 성능 저하원인 분석

### Architecture
- Overview
	![image](https://user-images.githubusercontent.com/41335296/132103298-161600bc-c6e3-4517-8eef-f35b69bd9727.png)

- vit_base_patch16_384
	![image](https://user-images.githubusercontent.com/26565404/139758414-1c1a4ed5-bf2a-4a64-9855-f75cf41257e5.png)
	
- Efficientnet_b3
	![image](https://user-images.githubusercontent.com/26565404/139758630-5ef4581a-0171-42be-9978-045c11e4f126.png)


### Requirements
* Python >= 3.5 (3.8 recommended)
* PyTorch >= 1.4 (1.9 recommended)
* tqdm (Optional for `test.py`)
* efficientnet_pytorch
* albumnetations
* torchmetrics
* timm
* wandb


### Folder Structure
  ```
	input/data/
	├──train
	│   ├── images
	│   │    ├ 000001_female_Asian_45
	│   │    │  ├ mask1.jpg
	│   │    │  ├ mask2.jpg
	│   │    │  ├ mask3.jpg
	│   │    │  ├ mask4.jpg
	│   │    │  ├ mask5.jpg
	│   │    │  ├ incorrect.jpg
	│   │    │  └ normal.jpg                          
	│   │    └ ...
	│   ├── train.csv
	│   └── trans_train.csv  
	└──eval
	├── info.csv
	└── images
	     ├── ._0a2bd33bf76d7426f3d6ca0b7fbe03ee431159b4.jpg
	     └── ...
	image-classification-level1-04/
	│
	├── train.py - main script to start training
	├── test.py - evaluation of trained model
	│
	├── config.json - holds configuration for training
	├── config_submit.json - holds configuration for submit
	│
	├── parse_config.py - class to handle config file and cli options
	│  
	├── base/ - abstract base classes
	│   ├── base_data_loader.py
	│   ├── base_model.py
	│   └── base_trainer.py
	│
	├── data_loader/ - anything about data loading goes here
	│   ├── data_loader.py
	│   ├── datasets.py
	│   └── transforms.py
	│
	├── model/ - models, losses, and metrics
	│   ├── model.py
	│   ├── metric.py
	│   └── loss.py
	│
	├── saved/
	│   ├── models/ - trained models are saved here
	│   └── log/ - default logdir for tensorboard and logging output
	│
	├── trainer/ - trainers
	│   └── trainer.py
	│
	├── logger/ - module for tensorboard visualization and logging
	│   ├── visualization.py
	│   ├── logger.py
	│   └── logger_config.json
	│  
	└── utils/ - small utility functions
	├── util.py
	└── ...
```

### How to use this Code

```python
# BaseModel
python3 train.py -c config_main.json
# AgeModel
python3 train.py -c config_sub.json

# Submit
python3 submit.py -c config_submit_main.json
python3 submit.py -c config_submit_sub.json
```

### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "PretrainModelTimmViTBasePath16_TRNS004_Adam_StepLR",  // training session name
    "n_gpu": 1,  // number of GPUs to use for training.
    "arch": {
        "type": "PretrainModelTimmViTBasePath16",                  // name of model architecture to train
        "args": {}
    },
    "transforms_select": {
        "type": "transforms_select",                               // selecting transforms methods
        "args": {
            "method": "VIT_TRNS004",
            "default": "VIT_DEFAULT"
        }
    },
    "data_loader": {
        "type": "MaskDataLoader",         			   // selecting data loader                
        "args": {
            "data_dir": "../input/data",  		    	   // dataset path
            "batch_size": 32,             			   // batch size
			"shuffle": true,                           // shuffle training data before splitting
			"validation_split": 0.1          	   // size of validation dataset. float(portion) or int(number of samples)
			"num_workers": 2,                	   // number of cpu processes to be used for data loading
			"trsfm": false,              		   // use transforms
			"submit": false                		   // submission

        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 2e-5,                     			   // learning rate
            "weight_decay": 0,                			   // (optional) weight decay
            "amsgrad": true
        }
    },
    "loss": {
        "type": "cross_entropy_loss",				   //loss
        "args": {
            "class_weight": false
        }
    },
    "metrics": [
        "accuracy","f1"						   // list of metrics to evaluate
    ],
    "lr_scheduler": {
        "type": "StepLR",					   // learning rate scheduler
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
		"epochs": 20,                                      // number of training epochs
		"save_dir": "saved/",              		   // checkpoints are saved in save_dir/models/name
		"save_period": 1,                    		   // save checkpoints every save_freq epochs
		"verbosity": 2,                    		   // 0: quiet, 1: per epoch, 2: full	
		"monitor": "min val_loss"          		   // mode and metric for model performance monitoring. set 'off' to disable.
		"early_stop": 10,	                	   // number of epochs to wait before early stop. set 0 to disable.  		
    },
    "wandb": {
        "use": true,						   //enable tensorboard visualization
        "args": {
            "project": "basic", 				   //sub project name
            "entity": "boostcamp-level01-04"			   //project name
        }
    }
}
```

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume (e.g. saved/models/[confg.name]/[MMDD_Hashvalue]/checkpoint-epoch#.pth)
  ```  

### Test from checkpoints
You can test from a previously saved checkpoint by:

  ```
  python test.py --resume path/to/checkpoint(e.g. saved/models/[confg.name]/[MMDD_Hashvalue]/checkpoint-epoch#.pth)
  ```

### Submit from checkpoints
You can submit from a previously saved checkpoint by:

  ```
  python submit.py --resume path/to/checkpoint(e.g. saved/models/[confg.name]/[MMDD_Hashvalue]/checkpoint-epoch#.pth)
  ```


## TODOs
- [x] Add transforms feature(or Albumentation)
- [x] `Weights & Biases` logger support
- [x] Add pretrained model(e.g. EfficientNet, ViT, Resnet ...)
- [x] Data Augmentation (e.g. CLAHE, Elastic & Cutmix, Horizontal Flip, ...)
- [x] Soft, Hard Ensemble
- [x] Time Test Augment
- [x] Multi Sample Dropout
- [x] Focal Loss
- [x] ArcFace Loss
- [x] Label Smoothing Loss
- [X] Angular Additive Margin Loss
- [x] OverSampling
- [x] K-Fold Validate
- [x] Crop using MTCNN


## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
- We use this project template([PyTorch-template](https://github.com/victoresque/pytorch-template)) by [victoresque](https://github.com/victoresque)


- [Imbalanced Dataset Sampler](https://github.com/ufoym/imbalanced-dataset-sampler)
- [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)
- **[All-Age-Faces-Dataset](https://github.com/JingchunCheng/All-Age-Faces-Dataset)**
