import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
import json
import natsort
import mrcfile
import random
import scipy
import scipy.ndimage
from lib.utils.multi import run_iterator

datapath = '/home/myz/inf_10'
classes = "6t3e,3gl1,1yg6,1f1b,2byu,4d4r"
class NewDataSet(Dataset):
    def __init__(self):
        #file = '/ldap_shared/shared/usr/xiangruz/clustering/test_30_003.pickle'

        #with open(file, 'rb') as f:
        #    x_train = pickle.load(f, encoding='latin1')
        #    x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, 5000)]), 1)

        #gt = list(range(5)) * 1000

        #gt.sort()
        #gt = np.array(gt)
        # self.x_data = (x_train - np.min(x_train)) / (np.max(x_train)-np.min(x_train))
        #self.x_data = x_train
        #print(np.max(self.x_data))
        #print(np.min(self.x_data))
        #self.y_data = gt
        #self.lenth = x_train.shape[0]
        #print(x_train.shape)
        
        #datapath = '/home/myz/binary' 
        root_dir = os.path.join(datapath, 'subtomogram_mrc')
        json_dir = os.path.join(datapath, 'json_label')
        self.root_dir = root_dir
        self.json_dir = json_dir
        all_imgs = os.listdir(root_dir)
        all_jsons = os.listdir(json_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.total_jsons = natsort.natsorted(all_jsons)
        print(f'{len(self.total_imgs)}, vs {len(self.total_jsons)}')
        assert( len(self.total_imgs) == len(self.total_jsons) )
        self.label_to_target = {} # 1bxn,2h12,    2ldb,3hhb
        for i, mol in enumerate(classes.split(",")):
            #for i, mol in enumerate("1bxn,2h12".split(",")):
            self.label_to_target[mol] = i

    def __getitem__(self, idx):
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

        with open(path_json) as f:
            MRC_dict = json.load(f)

        target = self.label_to_target[MRC_dict['name']]
        #if self.transform is not None:
        #    transformed_MRC_img = self.transform(MRC_img)
        #else:
        #    transformed_MRC_img = MRC_img
        transformed_MRC_img = random_rotation_3d(MRC_img, 180)

        transformed_MRC_img = move(transformed_MRC_img)
        
        return transformed_MRC_img, target
        #return x, self.y_data[index]


    def __len__(self):
        return len(self.total_jsons)

def move(image, size=32, h=40, w=40, d=40):
    new_d = size
    new_h = size
    new_w = size
    image_size = image.shape
    y = np.random.randint(new_h - h, h - new_h)
    x = np.random.randint(new_w - w, w - new_w)
    z = np.random.randint(new_d - d, d - new_d)
    image = scipy.ndimage.interpolation.shift(np.squeeze(image), [x, y, z])

    return image.reshape(image_size)

def random_rotation_3d(image, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """
    x = np.random.randint(0, 3)
    angle = random.uniform(-max_angle, max_angle)
    # angle2 = random.uniform(-max_angle, max_angle)
    # angle3 = random.uniform(-max_angle, max_angle)
    size = image.shape
    # print(size)
    image = np.squeeze(image)
    # print(image1.shape)  # (40,40,40)
    # image1 = batch[i]

    if x == 0:
        # rotate along z-axis
        image = scipy.ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
    elif x == 1:
        # rotate along y-axis
        image = scipy.ndimage.rotate(image, angle, axes=(0, 2), reshape=False)
    else:
        # rotate along x-axis
        image = scipy.ndimage.rotate(image, angle, axes=(1, 2), reshape=False)
    '''

    # rotate along z-axis
    image = scipy.ndimage.rotate(image, angle1, axes=(0, 1), reshape=False)
    # rotate along y-axis
    image = scipy.ndimage.rotate(image, angle2, axes=(0, 2), reshape=False)
    # rotate along x-axis
    image = scipy.ndimage.rotate(image, angle3, axes=(1, 2), reshape=False)
    '''

    return image.reshape(size)


class NewDataSet_test(Dataset):
    def __init__(self):
        root_dir = os.path.join(datapath, 'subtomogram_mrc')
        json_dir = os.path.join(datapath, 'json_label')
        self.root_dir = root_dir
        self.json_dir = json_dir
        all_imgs = os.listdir(root_dir)
        all_jsons = os.listdir(json_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.total_jsons = natsort.natsorted(all_jsons)
        print(f'{len(self.total_imgs)}, vs {len(self.total_jsons)}')
        assert( len(self.total_imgs) == len(self.total_jsons) )
        self.label_to_target = {}
        for i, mol in enumerate(classes.split(",")):
            self.label_to_target[mol] = i

    def __getitem__(self, idx):
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

        with open(path_json) as f:
            MRC_dict = json.load(f)

        target = self.label_to_target[MRC_dict['name']]
        return MRC_img, target

    def __len__(self):
        return len(self.total_jsons)
