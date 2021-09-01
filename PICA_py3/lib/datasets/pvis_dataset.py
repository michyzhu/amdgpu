import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import random
import scipy
import scipy.ndimage
from lib.utils.multi import run_iterator
file = '/home/myz/10_2000_30_01.pickle'
classes = ['1I6V', '1QO1', '3DY4','4V4A','5LQW']#, '1KP8', '1A1S','1BXR','1VPX','1LB3']
class NewDataSet(Dataset):
    def __init__(self):
        '''
        file = '/ldap_shared/shared/usr/xiangruz/clustering/2000_30_001.pickle'
        with open(file, 'rb') as f:
            x_train = pickle.load(f, encoding='latin1')
            x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, 20000)]), 1)

        gt = list(range(10)) * 2000  # (20000,)   [0,0,0,0,0,......,9,9,9,9]
        '''

        #file = '/ldap_shared/shared/usr/xiangruz/clustering/test_30_003.pickle'
        with open(file, 'rb') as f:
            x_train = pickle.load(f, encoding='latin1')
            x_train = np.array([x_train[_]['v'] for _ in range(0, len(x_train), 1) if x_train[_]['id'] in classes])
            #x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, len(x_train), 1) if x_train[_]['id'] in classes]),-1)
            #x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, 5000)]), -1)

        #gt = list(range(5)) * 1000
        #gt.sort()
        #gt = np.array(gt)
        gt = np.repeat(range(len(classes)), 2000)  #ground truth labels

        # self.x_data = (x_train - np.min(x_train)) / (np.max(x_train)-np.min(x_train))
        self.x_data = x_train
        print(np.max(self.x_data))
        print(np.min(self.x_data))
        self.y_data = gt
        self.lenth = x_train.shape[0]
        print(x_train.shape)

    def __getitem__(self, index):
        x = zoom(np.expand_dims(self.x_data[index],0))
        x = random_rotation_3d(x, 180)
        x = move(x)
        #print(f'trans: SHAPE:{x.shape}, TYPE: {type(x)}')
        return x, self.y_data[index]


    def __len__(self):
        return self.lenth

def zoom(image):
    f = 32.0 / 24
    new = scipy.ndimage.zoom(image, [1,f,f,f])
    return new


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
        '''
        file = '/ldap_shared/shared/usr/xiangruz/clustering/2000_30_001.pickle'
        with open(file, 'rb') as f:
            x_train = pickle.load(f, encoding='latin1')
            x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, 20000)]), 1)

        gt = list(range(10)) * 2000  # (20000,)   [0,0,0,0,0,......,9,9,9,9]
        '''

        #file = '/ldap_shared/shared/usr/xiangruz/clustering/test_30_003.pickle'

        with open(file, 'rb') as f:
            x_train = pickle.load(f, encoding='latin1')
            print(f'LENGTH XTRAIN: {len(x_train)}')
            x_train = np.array([x_train[_]['v'] for _ in range(0, len(x_train), 1) if x_train[_]['id'] in classes and _ % 2000 < 100])
            #x_train = np.array([x_train[_]['v'] for _ in range(0, 5000) if x_train[_]['id'] in classes])

            #x_train = np.expand_dims(np.array([x_train[_]['v'] for _ in range(0, 5000) if x_train[_]['id'] in classes]), -1)
        #gt = list(range(2)) * 1000

        #gt.sort()
        #gt = np.array(gt)
        # self.x_data = (x_train - np.min(x_train)) / (np.max(x_train)-np.min(x_train))
        gt = np.repeat(range(len(classes)), 100)
        self.x_data = x_train
        print(np.max(self.x_data))
        print(np.min(self.x_data))
        self.y_data = gt
        self.lenth = x_train.shape[0]
        print(x_train.shape)

    def __getitem__(self, index):
        x = zoom(np.expand_dims(self.x_data[index],0))
        #print(f'orig: SHAPE:{x.shape}, type: {type(x)}')
        return x, self.y_data[index]

    def __len__(self):
        return self.lenth
