# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import copy

from data_providers.base_provider import *

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len_1 = len(datasets[0])
        self.len_2 = len(datasets[1])

    def __getitem__(self, i):
        inputs, targets = self.datasets[0][i]
        arch_inputs, arch_targets = self.datasets[1][i%self.len_2]
        return inputs, targets, arch_inputs, arch_targets

    def __len__(self):
        return max(len(d) for d in self.datasets)


class imagenet_100DataProvider(DataProvider):

    def __init__(self, dataset_location=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = dataset_location
        self.train_batch_size = train_batch_size
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        train_dataset = datasets.ImageFolder(self.train_path, train_transforms)
        length = len(train_dataset)
        indices = list(range(length))
        samples = train_dataset.samples
        random.shuffle(samples)
        samples_1, samples_2 = samples[:length//6*5], samples[length//6*5:]
        training_data_1, training_data_2 = copy.deepcopy(train_dataset), copy.deepcopy(train_dataset)
        training_data_1.samples, training_data_2.samples = samples_1, samples_2
        training_data_dual = ConcatDataset(training_data_1, training_data_2)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            valid_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ]))

            self.train = torch.utils.data.DataLoader(
                training_data_dual, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.valid_path, transforms.Compose([
                transforms.Resize(self.resize_value),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                self.normalize,
            ])), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet_100'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            raise ValueError('unable to access imagenet_100')
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download imagenet_100')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                color_transform,
                transforms.ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 224