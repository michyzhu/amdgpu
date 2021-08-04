# https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/2
# https://github.com/xulabs/projects/blob/master/autoencoder/autoencoder_util.py
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import natsort
import mrcfile
import json
import random

import data_config


# random.seed(1)
# torch.manual_seed(1)

def mapping_types(num_classes):
    if num_classes == 10:
        labels = data_config.class_10
    elif num_classes == 50:
        labels = data_config.class_50
    elif num_classes == 100:
        labels = data_config.class_100
    else:
        raise ValueError("num_classes only support (10, 50, 100)")

    label_to_target = {label: idx for idx, label in enumerate(labels)}
    return label_to_target


class CryoETDatasetLoader(Dataset):
    def __init__(self, root_dir, json_dir, transform=None):
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.transform = transform
        all_imgs = os.listdir(root_dir)
        all_jsons = os.listdir(json_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.total_jsons = natsort.natsorted(all_jsons)
        print(f'{len(self.total_imgs)}, vs {len(self.total_jsons)}')
        assert (len(self.total_imgs) == len(self.total_jsons))
        num_classes = 10
        # {"1bxn": 0, "1f1b": 1, "1yg6": 2, "2byu": 3, "2h12": 4, "2ldb": 5, "3gl1": 6, ...}
        self.label_to_target = mapping_types(num_classes)

    def __len__(self):
        return len(self.total_jsons)

    def __getitem__(self, idx):
        # mrc_img = mrc_img.astype(np.float32).transpose((2,1,0)).reshape((1,28,28,28))
        path_img = os.path.join(self.root_dir, self.total_imgs[idx])
        path_json = os.path.join(self.json_dir, self.total_jsons[idx])

        with mrcfile.open(path_img, mode='r+', permissive=True) as mrc:
            mrc_img = mrc.data
            if mrc_img is None:
                print(path_img)
            try:
                mrc_img = mrc_img.astype(np.float32).transpose((2, 1, 0)).reshape((1, 32, 32, 32))
            except:
                print(mrc_img.shape)
                print(path_img)

        # mean = 0.06968536
        # std = 0.12198435
        # mrc_img = (mrc_img - mean) / std

        with open(path_json) as f:
            mrc_json = json.load(f)

        target = self.label_to_target[mrc_json['name']]

        if self.transform is not None:
            transformed_mrc_img = self.transform(mrc_img)
        else:
            transformed_mrc_img = mrc_img

        return transformed_mrc_img, target


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


'''
import torchio as tio

augmentation = [
            tio.transforms.RandomFlip(),
            tio.transforms.RandomBlur(),
            tio.transforms.RandomAffine(),
            tio.transforms.ZNormalization()
        ]

train_dataset = CryoETDatasetLoader('/shared/home/c_myz/data/data3_SNRinfinity/subtomogram_mrc', '/shared/home/c_myz/data/data3_SNRinfinity/json_label',

            transform =
            transforms.Compose(augmentation))


train_loader = torch.utils.data.DataLoader(train_dataset)

import Encoder3D.Model_RB3D

model = Encoder3D.Model_RB3D.RB3D()

for i, (images, target) in enumerate(train_loader):
    print(i, images, target)
    print(len(images))
    print(images[0].shape)#,images[1].shape)
    print(type(images[0]))
    print(target)
    o = model(images)
    print(o)
    print(torch.max(images))
    print(torch.min(images))
    break
    '''
