import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle        
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)    

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        '''
        stratified train & val split
        '''
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        labels = self.dataset.df[:, -1]

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert self.n_samples * split < self.n_samples, "validation set size is configured to be larger than entire dataset."
        
        train_idx, valid_idx = train_test_split(idx_full, stratify=labels, test_size=split, random_state=42)
        
        train_sampler = Sampler(train_idx)
        valid_sampler = Sampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
