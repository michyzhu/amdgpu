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

# random.seed(1)
# torch.manual_seed(1)

class CryoETDatasetLoader(Dataset):
    def __init__(self, root_dir, json_dir, transform=None):
	#pickle_name,
	#final = pickley(currPath)	
    	#data = final['data']
        self.root_dir = root_dir
        self.json_dir = json_dir
        self.transform = transform
        all_imgs = os.listdir(root_dir)
        all_jsons = os.listdir(json_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.total_jsons = natsort.natsorted(all_jsons)
        print(f'{len(self.total_imgs)}, vs {len(self.total_jsons)}')
        assert( len(self.total_imgs) == len(self.total_jsons) )
        # self.imgLabelPairs = data
        self.labeled = 0
        self.label_to_target = {}
        # self.label_to_target = {"31": 0, "33": 1, "35": 2, "43": 3, "69": 4, "72": 5, "73": 6}

    def __len__(self):
        return len(self.total_jsons)        
	#return len(self.imgLabelPairs)

    def __getitem__(self, idx):
	# img = self.imgLabelPairs[idx]['v']        
	# label = self.imgLabelPairs[idx]['id']	
        # MRC_img = MRC_img.astype(np.float32).transpose((2,1,0)).reshape((1,28,28,28))
        path_img = os.path.join(self.root_dir, self.total_imgs[idx])
        path_json = os.path.join(self.json_dir, self.total_jsons[idx])

        with mrcfile.open(path_img, mode='r+', permissive=True) as mrc:
            MRC_img = mrc.data
            if(MRC_img is None):
                print(path_img)
            try:
                MRC_img = MRC_img.astype(np.float32).transpose((2,1,0)).reshape((1,32,32,32)) 
            except:
                print(MRC_img.shape)
                print(path_img)

        # mean = 0.06968536 
        # std = 0.12198435
        # MRC_img = (MRC_img - mean) / std
       
        with open(path_json) as f:
            MRC_dict = json.load(f)
        
        if(self.label_to_target.get(MRC_dict['name']) == None):
            self.label_to_target[MRC_dict['name']] = self.labeled
            self.labeled+=1
        target = self.label_to_target[MRC_dict['name']]

        if self.transform is not None:
            transformed_MRC_img = self.transform(MRC_img)
        else:
            transformed_MRC_img = MRC_img
        #print(target)

        return transformed_MRC_img, target



#Checking -> Wokring fine



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
