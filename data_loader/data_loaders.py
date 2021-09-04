import os
from .datasets import MakeDataset, MaskSubmitDataset, MakeAgeDataset, BasicDataset
from .transforms import transforms_select
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

#[TODO] Faster DataLoader
class MaskDataLoader():
    def __init__(self, data_dir, batch_size, shuffle=True,
                validation_split=0.1, num_workers=1, sampler=None,
                trsfm=None, default_trsfm=None, submit=False, is_main=None):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')
        
        assert trsfm is not None, "Set the transform on train set"
        assert default_trsfm is not None, "Set the default transform"

        self.trsfm = trsfm
        self.default_trsfm = default_trsfm # if you use VIT, use VIT_DEFAULT
        print(f'Current transforms : {self.trsfm}')
        print('num_workers: ', num_workers)

        if trsfm is None: 
            self.trsfm = self.default_trsfm # if you use VIT, use VIT_DEFAULT
        
        if is_main:
            self.dataset = MakeDataset(csv_path=os.path.join(self.train_dir, 'train.csv'),
                                        trsfm=self.trsfm, defalut_trsfm=self.default_trsfm)
        else:
            self.dataset = MakeAgeDataset(csv_path=os.path.join(self.train_dir, 'train.csv'),
                                        trsfm=self.trsfm, defalut_trsfm=self.default_trsfm)
            
        if validation_split < 0:
            raise 'Validation Split ratio < 0'

        # Default_trsfrm is applied to valid_datset whether you set up self.trsfm or not
        self.train_dataset, self.valid_dataset = self.dataset.split_validation(validation_split)

        if sampler == 'over':
            print('Apply Oversampling, Turn off the shuffle')
            self.shuffle = False
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=self.shuffle,
                                        num_workers=num_workers, sampler=ImbalancedDatasetSampler(self.train_dataset), pin_memory=True)
        elif sampler == 'normal':
            print('No sampling method')
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=self.shuffle,
                                        num_workers=num_workers, pin_memory=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size, shuffle=False,
                                        num_workers=num_workers, sampler=None, pin_memory=True)

        if submit:
            self.submit_dataset = MaskSubmitDataset(transform=self.default_trsfm)
            self.submit_dataloader = DataLoader(self.submit_dataset, batch_size, shuffle=False,
                                            num_workers=num_workers, sampler=None, pin_memory=True)
        else:
            self.submit_dataset = None
            self.submit_dataloader = None

    def split_validation(self):
        return self.train_dataloader, self.valid_dataloader, self.submit_dataloader