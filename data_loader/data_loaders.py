import os
from .datasets import MaskDataset, MaskSubmitDataset
from .transforms import transforms_select
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

#[TODO] Faster DataLoader
class MaskDataLoader():
    def __init__(self, data_dir, batch_size, shuffle=True,
                validation_split=0.1, num_workers=1, sampler=None,
                trsfm=None, submit=False):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'eval')
        self.trsfm = trsfm
        self.default_trsfm = transforms_select(method='DEFAULT') # if you use VIT, use VIT_DEFAULT
        print(f'Current transforms : {self.trsfm}')
        print('num_workers: ', num_workers)

        if trsfm is None: 
            self.trsfm = self.default_trsfm # if you use VIT, use VIT_DEFAULT

        self.dataset = MaskDataset(csv_path=os.path.join(self.train_dir, 'train.csv'),
                                    transform=self.trsfm)

        if validation_split < 0:
            raise 'Validation Split ratio < 0'

        # Default_trsfrm is applied to valid_datset whether you set up self.trsfm or not
        self.train_dataset, self.valid_dataset = self.dataset.split_validation(validation_split,
                                                                valid_trsfm=self.default_trsfm)

        if sampler == 'over':
            print('Apply Oversampling, Turn off the shuffle')
            self.shuffle = False
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=self.shuffle,
                                        num_workers=num_workers, sampler=ImbalancedDatasetSampler(self.train_dataset), pin_memory=True)
        else:
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

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     dataLoader = MaskDataLoader(data_dir="../input/data", batch_size=16)
#     images, labels = next(iter(dataLoader))
#     plt.figure(figsize=(12,12))
#     for n, (image, label) in enumerate(zip(images, labels), start=1):
#         plt.subplot(4,4,n)
#         image=image.permute(1,2,0)
#         plt.imshow(image)
#         plt.title("{}".format(label))
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
