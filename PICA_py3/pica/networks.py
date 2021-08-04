#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Raymond Huang (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci/PICA

import torch.nn as nn
import torch
from lib import Config as cfg
from lib.networks import DefaultModel, Flatten, register
from lib.utils.loggers import STDLogger as logger

__all__ = ['ResNet34', 'YOPO']

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
  """3x3 convolution with padding"""
  return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=padding, bias=False, groups=groups)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=None, num_group=32):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
    self.celu = nn.CELU(alpha=0.075, inplace=True)
    self.conv2 = conv3x3(planes, planes, groups=num_group)
    self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.celu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.celu(out)

    return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=None, num_group=32):
        super(Bottleneck, self).__init__()
        mid_planes = num_group * planes // 32
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm3d(mid_planes, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv3d(mid_planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet34(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')
        cfg.add_argument('--net-avgpool-size', default=3, type=int, choices=[3, 5, 7],
                        help='Avgpool kernel size determined by inputs size')

    def __init__(self, cin, cout, sobel, net_heads=None, pool_size=None, num_group=32):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        pool_size = pool_size if pool_size is not None else cfg.net_avgpool_size
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(ResNet34, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 64

        self.layer1 = nn.Sequential(nn.Conv3d(2 if sobel else cin, 64,
                    kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(64, track_running_stats=True),
                    nn.CELU(alpha=0.075, inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2, padding=1))

        self.layer2 = self._make_layer(BasicBlock, 64, 3, num_group)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, num_group, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 6, num_group, stride=2)
        self.layer5 = self._make_layer(BasicBlock, 512, 3, num_group, stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool3d(pool_size, stride=1), Flatten())
        heads = [nn.Sequential(nn.Linear(512 * BasicBlock.expansion, head),
                               nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

          downsample = nn.Sequential(
              nn.Conv3d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm3d(planes * block.expansion,
                             track_running_stats=True)
          )
          

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                                    track_running_stats=True, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                    track_running_stats=True, num_group=num_group))

        return nn.Sequential(*layers)

    def run(self, x, target=None):
        """Function for getting the outputs of intermediate layers
        """
        if target is None or target > 5:
            raise NotImplementedError('Target is expected to be smaller than 6')
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        for layer in layers[:target]:
            x = layer(x)
        return x

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return list(map(lambda head:head(x), self.heads))

class ResNext50(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')
        cfg.add_argument('--net-avgpool-size', default=3, type=int, choices=[3, 5, 7],
                        help='Avgpool kernel size determined by inputs size')

    def __init__(self, cin, cout, sobel, net_heads=None, pool_size=None, num_group=32):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        pool_size = pool_size if pool_size is not None else cfg.net_avgpool_size
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(ResNext50, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 64

        self.layer1 = nn.Sequential(nn.Conv3d(2 if sobel else cin, 64,
                    kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm3d(64, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2, padding=1))

        self.layer2 = self._make_layer(Bottleneck, 64, 3, num_group)
        self.layer3 = self._make_layer(Bottleneck, 128, 4, num_group, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, 6, num_group, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 512, 3, num_group, stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool3d(pool_size, stride=1), Flatten())
        heads = [nn.Sequential(nn.Linear(512 * Bottleneck.expansion, head),
            nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
            nn.Conv3d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes * block.expansion,
                        track_running_stats=True)
          )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                                    track_running_stats=True, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                    track_running_stats=True, num_group=num_group))

        return nn.Sequential(*layers)

    def run(self, x, target=None):
        """Function for getting the outputs of intermediate layers
        """
        if target is None or target > 5:
            raise NotImplementedError('Target is expected to be smaller than 6')
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        for layer in layers[:target]:
            x = layer(x)
        return x

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return list(map(lambda head:head(x), self.heads))


class Block(nn.Module):

  def __init__(self, inplanes, planes, poolsize, stride=2, track_running_stats=None):
    super(Block, self).__init__()

    assert (track_running_stats is not None)

    self.conv = conv3x3(inplanes, planes, stride, padding=0)
    self.bn = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
    self.elu = nn.ELU(inplace=True)
    self.pool = nn.MaxPool3d(kernel_size=poolsize, stride=1)

  def forward(self, x):

    x = self.conv(x)
    x = self.bn(x)
    x = self.elu(x)
    out = self.pool(x)

    return x, out


class YOPO(DefaultModel):

    @staticmethod
    def require_args():

        cfg.add_argument('--net-heads', nargs='*', type=int,
                        help='net heads')

    def __init__(self, cin, cout, sobel, net_heads=None, pool_size=None):
        net_heads = net_heads if net_heads is not None else cfg.net_heads
        logger.debug('Backbone will be created wit the following heads: %s' % net_heads)
        # do init
        super(YOPO, self).__init__()
        # build sobel
        self.sobel = self._make_sobel_() if sobel else None
        # build trunk net
        self.inplanes = 2 if sobel else cin
        self.layer1 = Block(self.inplanes, 128, poolsize=19, track_running_stats=True)
        self.layer2 = Block(128, 128, poolsize=9, track_running_stats=True)
        self.layer3 = Block(128, 128, poolsize=4, track_running_stats=True)
        self.layer4 = Block(128, 128, poolsize=1, track_running_stats=True)
        # self.layer5 = Block(128, 128, poolsize=20, track_running_stats=True)
        # self.layer6 = Block(128, 128, 2, track_running_stats=True)
        # self.layer7 = Block(128, 128, 2, track_running_stats=True)
        # self.layer8 = Block(128, 128, 2, track_running_stats=True)

        self.layer9 = nn.Sequential(nn.Dropout(p=0.5),
                                    Flatten(),
                                    nn.Linear(128*4, 512),
                                    nn.ELU(0.3),
                                    nn.BatchNorm1d(512),
                                    nn.Linear(512, 256),
                                    nn.ELU(0.3),
                                    nn.BatchNorm1d(256),
                                    nn.Linear(256, 64)
                                    )
        '''
        self.layer9 = nn.Sequential(nn.Dropout(p=0.5), Flatten())
        self.layer10 = nn.Linear(512, 512)
        self.layer11 = nn.ELU(0.3)
        self.layer12 = nn.BatchNorm1d(512)
        self.layer13 = nn.Linear(512, 256)
        self.layer14 = nn.ELU(0.3)
        self.layer15 = nn.BatchNorm1d(256)
        self.layer16 = nn.Linear(256, 64)
        '''
        heads = [nn.Sequential(nn.Linear(64, head),  nn.Softmax(dim=1)) for head in net_heads]
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        if self.sobel is not None:
            x = self.sobel(x)
        x, m1 = self.layer1(x)
        x, m2 = self.layer2(x)
        x, m3 = self.layer3(x)
        x, m4 = self.layer4(x)
        # x, m5 = self.layer5(x)
        # x, m6 = self.layer6(x)
        # x, m7 = self.layer7(x)
        # x, m8 = self.layer8(x)
        m = self.layer9(torch.cat([m1,m2,m3,m4], axis=1))
        '''
        m = self.layer10(m)
        m = self.layer11(m)
        m = self.layer12(m)
        m = self.layer13(m)
        m = self.layer14(m)
        m = self.layer15(m)
        m = self.layer16(m)
        '''
        return list(map(lambda head:head(m), self.heads))



register('resnet34', ResNet34)
register('YOPO', YOPO)
