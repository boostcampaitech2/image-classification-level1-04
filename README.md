# Boostcamp AI-Tech 2nd - Master Slave Team

Level 1 P-Stage Image Classification Project Repository

## Members
* 김대웅, 나요한, 박준수, 이호영, 추창한, 최한준, 한진

## Requirements
* Python >= 3.5 (3.8 recommended)
* PyTorch >= 1.4 (1.9 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* efficientnet_pytorch

## Folder Structure
  ```
  input/data/
  │        ├──train/...
  │        └──eval/...
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

## Usage
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "PretrainResNet18_Test",      // training session name
  "n_gpu": 1,                           // number of GPUs to use for training.
  
  "arch": {
    "type": "PretrainModelTV",          // name of model architecture to train
    "args": {}                
  },

  "transforms_select": {
      "type": "transforms_select",      // seleecting transforms methods
      "args": {
          "method": "DEFAULT"
      }
  },
    
  "data_loader": {
    "type": "MaskDataLoader",          // selecting data loader
    "args":{
      "data_dir": "../input/data",     // dataset path
      "batch_size": 128,               // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
      "trsfm": false,                  // use transforms
      "submit": false                  // submission

    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },

  "loss": {
      "type": "cross_entropy_loss",    // loss
      "args":{
          "class_weight": true
      }
  },
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
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
  

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint(e.g. saved/models/[confg.name]/[MMDD_Hashvalue]/checkpoint-epoch#.pth)
  ```

## Customization

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.

## TODOs

- [x] Add transforms feature(or Albumentation)
- [ ] `Weights & Biases` logger support
- [x] Add pretrained model(e.g. , EfficientNet, ...)
- [ ] Add Raytune Module

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
- We use this project template([PyTorch-template](https://github.com/victoresque/pytorch-template)) by [victoresque](https://github.com/victoresque)
