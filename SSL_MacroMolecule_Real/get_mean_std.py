#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FileName:	get_mean_std.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:51:36
"""

import os
import numpy as np
import torchvision.transforms as transforms
import pickle

import Custom_CryoET_DataLoader
from CustomTransforms import ToTensor


class Dataloader:
    def __init__(self, root='./Datasets/'):
        self.dirs = ['train', 'val']

        self.means = [0]
        self.stds = [0]

        self.dataset = {x: Custom_CryoET_DataLoader.CryoETDatasetLoader(
                root_dir=os.path.join(root, x, 'subtomogram_mrc'), json_dir=os.path.join(root, x, 'json_label'),
                transform=None) for x in self.dirs}

    def get_mean_std(self, type, mean_std_path):
        """
        calculate the mean and std of the dataset
        :param type: datatype, such as 'train', 'val'
        :param mean_std_path: saved path
        :return:
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            mrc_img = data[0]
            self.means[0] += mrc_img[0, :, :, :].mean()
            self.stds[0] += mrc_img[0, :, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stds = np.asarray(self.stds) / num_imgs

        print("{} : NormMean = {}".format(type, self.means))
        print("{} : NormStds = {}".format(type, self.stds))

        with open(mean_std_path, 'wb') as f:
            pickle.dump(self.means, f)
            pickle.dump(self.stds, f)
            print('pickle done')


if __name__ == "__main__":
    root_path = "./Datasets/"
    dataloader = Dataloader(root=root_path)
    for x in dataloader.dirs:
        mean_std_path = root_path + 'mean_std_value_' + x + '.pkl'
        dataloader.get_mean_std(x, mean_std_path)
