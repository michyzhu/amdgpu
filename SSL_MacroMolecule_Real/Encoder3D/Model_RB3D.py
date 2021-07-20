# -*- coding: utf-8 -*-
"""RB3D.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Aw7LhLKzynpA5y7zhA229INEmA_G2LC6

sources used to make this code:
https://github.com/pytorch/vision/blob/e6b4078ec73c2cf4fd4432e19c782db58719fb99/torchvision/models/resnet.py
https://jarvislabs.ai/blogs/resnet
"""

import torch
import torch.nn as nn
import random

random.seed(1)
torch.manual_seed(1)

class Bottleneck(nn.Module):

    def __init__(self):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(32, 16, kernel_size=3, stride=1,
                     padding=1, groups=1)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=1, stride=1,
                     padding=0, groups=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1,
                     padding=1, groups=1)
        self.conv4 = nn.Conv3d(32, 16, kernel_size=1, stride=1,
                     padding=0, groups=1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_R = self.conv1(x)
        out_R = self.relu(out_R)

        out_L = self.conv2(x)
        out_L = self.relu(out_L)

        out_L = self.conv3(out_L)
        out_L = self.relu(out_L)

        out_L = self.conv4(out_L)
        #out_L = self.relu(out_L)

        out_L = self.dropout(out_L)

        out = torch.cat((out_L, out_R), dim=1)
        out = self.relu(out)

        return out


class RB3D(nn.Module):
    def __init__(self, num_classes=7):
        super(RB3D, self).__init__()

        # assertion for Noble data
        # assert (num_classes == 7)
        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 32
        H = 32
        W = 32

        self.conv1 = nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.bottleneck_layers = nn.Sequential(*[Bottleneck(), Bottleneck(), Bottleneck(), Bottleneck()])

        self.fc1 = nn.Linear( (D*H*W*32) // 8 , 1024)
        self.fc2 = nn.Linear(1024 , 1024)
        self.fc3 = nn.Linear(1024 , num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        #self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bottleneck_layers(x)
        x = x.view(-1, (32*32*32*32) // 8)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        #x = self.softmax(x)

        return x



