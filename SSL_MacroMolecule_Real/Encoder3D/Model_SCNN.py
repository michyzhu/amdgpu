"""
FileName:	Model_SCNN.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:53:33
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()

        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 32
        H = 32
        W = 32

        self.conv1 = nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm_conv2 = nn.BatchNorm3d(num_features=32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm_conv4 = nn.BatchNorm3d(num_features=64)

        self.fc1 = nn.Linear( (D*H*W*64) // (8*8), 512)
        self.fc2 = nn.Linear(512 , 512)
        self.fc3 = nn.Linear(512 , num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm_conv2(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.batchnorm_conv4(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)

        x = x.view(-1, (32*32*32*64) // (8*8))

        x = F.relu(self.fc1(x))

        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        x = self.fc3(x)

        return x
