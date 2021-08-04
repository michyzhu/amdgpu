"""
FileName:	Model_RB3D.py
Author:	Zhu Zhan
Email:	henry664650770@gmail.com
Date:		2021-08-04 18:53:54
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(64, 32, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1, stride=1,
                               padding=0)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1,
                               padding=1)
        self.conv4 = nn.Conv3d(64, 32, kernel_size=1, stride=1,
                               padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 64 x 8 x 8 x 8 -> 32 x 8 x 8 x 8
        out_R = self.conv1(x)
        out_R = self.relu(out_R)
        # 64 x 8 x 8 x 8 -> 32 x 8 x 8 x 8
        out_L = self.conv2(x)
        out_L = self.relu(out_L)
        # 32 x 8 x 8 x 8 -> 64 x 8 x 8 x 8
        out_L = self.conv3(out_L)
        out_L = self.relu(out_L)
        # 64 x 8 x 8 x 8 -> 32 x 8 x 8 x 8
        out_L = self.conv4(out_L)
        # out_L = self.relu(out_L)
        out_L = self.dropout(out_L)
        # (32 x 8 x 8 x 8) + (32 x 8 x 8 x 8) -> 64 x 8 x 8 x 8
        out = torch.cat((out_L, out_R), dim=1)
        out = self.relu(out)

        return out


class RB3D(nn.Module):
    def __init__(self, num_classes=10):
        super(RB3D, self).__init__()
        # dimensions of the 3D image. Channels, Depth, Height, Width
        C = 1
        D = 32
        H = 32
        W = 32

        self.conv1 = nn.Conv3d(C, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.bottleneck_layers = nn.Sequential(*[Bottleneck(), Bottleneck(), Bottleneck(), Bottleneck()])
        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 1 x 32 x 32 x 32 -> 32 x 32 x 32 x 32
        x = self.conv1(x)
        x = self.relu(x)
        # 32 x 32 x 32 x 32 -> 32 x 16 x 16 x 16
        x = self.maxpool(x)
        # 32 x 16 x 16 x 16 -> 64 x 16 x 16 x 16
        x = self.conv2(x)
        x = self.relu(x)
        # 64 x 16 x 16 x 16 -> 64 x 8 x 8 x 8
        x = self.maxpool(x)
        # 64 x 8 x 8 x 8 -> 64 x 8 x 8 x 8
        x = self.bottleneck_layers(x)
        # 64 x 8 x 8 x 8 -> 64 x 2 x 2 x 2
        x = self.avgpool(x)
        # 64 x 2 x 2 x 2 -> 512
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        # 512 -> num_classes
        x = self.fc(x)

        return x
