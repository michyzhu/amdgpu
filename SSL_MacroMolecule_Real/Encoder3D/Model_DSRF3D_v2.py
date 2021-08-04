"""
FileName:	Model_DSRF3D_v2.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:53:45
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSRF3D_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(DSRF3D_v2, self).__init__()
        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 32
        H = 32
        W = 32

        self.conv1 = nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1 x 32 x 32 x 32 -> 32 x 32 x 32 x 32
        x = F.relu(self.conv1(x))
        # 32 x 32 x 32 x 32 -> 32 x 32 x 32 x 32
        x = F.relu(self.conv2(x))
        # 32 x 32 x 32 x 32 -> 32 x 16 x 16 x 16
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
        # 32 x 16 x 16 x 16 -> 64 x 16 x 16 x 16
        x = F.relu(self.conv3(x))
        # 64 x 16 x 16 x 16 -> 64 x 16 x 16 x 16
        x = F.relu(self.conv4(x))
        # 64 x 16 x 16 x 16 -> 64 x 8 x 8 x 8
        x = F.max_pool3d(x, kernel_size=2, stride=2, padding=0)
        # 64 x 8 x 8 x 8 -> 128 x 8 x 8 x 8
        x = F.relu(self.conv5(x))
        # 128 x 8 x 8 x 8 -> 128 x 8 x 8 x 8
        x = F.relu(self.conv6(x))
        # 128 x 8 x 8 x 8 -> 128 x 1 x 1 x 1
        x = self.avgpool(x)
        # 128 x 1 x 1 x 1 -> 128
        x = torch.flatten(x, 1)
        # # 128 -> num_classes
        x = self.fc(x)

        return x
