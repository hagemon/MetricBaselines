from PIL import Image
import torch.utils.data as data_utils
import os
import numpy as np
from torch.utils.data import DataLoader
from sampler import BalancedSampler


class Dataset(data_utils.Dataset):
    def __init__(self, file_name, root_dir, transform=None):
        lines = open(file_name).readlines()
        self.img_names = [l.strip().split()[0] for l in lines]
        # self.labels = np.array([[float(j) for j in l.strip().split()[1:]] for l in lines], dtype=np.float64)
        self.labels = np.array([int(l.strip().split()[1]) for l in lines])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.img_names[item])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[item]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample


def get_datasets(dataset, data_dir, transforms):
    image_folder = {'CUB': 'images'}
    image_datasets = {
        'train': Dataset(file_name=os.path.join(data_dir, 'train.txt'),
                         root_dir=os.path.join(data_dir, image_folder[dataset]),
                         transform=transforms['train']),
        'test': Dataset(file_name=os.path.join(data_dir, 'test.txt'),
                        root_dir=os.path.join(data_dir, image_folder[dataset]),
                        transform=transforms['test'])
    }
    return image_datasets


def get_data_loaders(datasets, batch_size, val_batch_size, n_instance, balanced=False, cm=None):
    balanced_sampler = BalancedSampler(datasets['train'], batch_size=batch_size, n_instance=n_instance)
    if cm is not None:
        train_loader = DataLoader(
            datasets['train'],
            num_workers=8,
            batch_sampler=cm,
            pin_memory=True,
        )
    elif balanced:
        train_loader = DataLoader(
            datasets['train'],
            num_workers=8,
            batch_sampler=cm if cm else balanced_sampler,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            datasets['train'],
            shuffle=True,
            batch_size=batch_size,
            num_workers=8,
        )
    dataset_loaders = {
        'train': train_loader,
        'mean': DataLoader(datasets['train'], batch_size=batch_size, num_workers=8),
        'test': DataLoader(datasets['test'], batch_size=val_batch_size, num_workers=8),
    }
    return dataset_loaders
