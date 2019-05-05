#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Sun Apr 21 2019 
@Time    : 17:55:59
@File    : resnet.py
@Author  : alpha
"""


import torch.nn as nn
import torch.nn.functional as F

from .utils import NormedConv2D, NormedLinear


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        self.downsample = downsample
        super(ResBlock, self).__init__()
        if self.downsample:
            self.branch1 = NormedConv2D(in_channels, out_channels, kernel_size=1, stride=2)
        self.branch2a = NormedConv2D(in_channels, out_channels // 4, kernel_size=1)
        self.branch2b = NormedConv2D(in_channels=out_channels // 4,
                                     out_channels=out_channels // 4,
                                     kernel_size=3,
                                     padding=1,
                                     stride=2 if self.downsample else 1)
        self.branch2c = NormedConv2D(out_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x) if self.downsample else x
        branch2a = F.relu_(self.branch2a(x))
        branch2b = F.relu_(self.branch2b(branch2a))
        branch2c = self.branch2c(branch2b)
        return branch1 + branch2c


class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super(ResNet50, self).__init__()
        self.conv1 = NormedConv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2a = ResBlock(64, 256)
        self.res2b = ResBlock(256, 256)
        self.res2c = ResBlock(256, 256)
        self.res3a = ResBlock(256, 512, downsample=True)
        self.res3b = ResBlock(512, 512)
        self.res3c = ResBlock(512, 512)
        self.res3d = ResBlock(512, 512)
        self.res4a = ResBlock(512, 1024, downsample=True)
        self.res4b = ResBlock(1024, 1024)
        self.res4c = ResBlock(1024, 1024)
        self.res4d = ResBlock(1024, 1024)
        self.res4e = ResBlock(1024, 1024)
        self.res4f = ResBlock(1024, 1024)
        self.res5a = ResBlock(1024, 2048, downsample=True)
        self.res5b = ResBlock(2048, 2048)
        self.res5c = ResBlock(2048, 2048)
        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = NormedLinear(2048, n_classes)

    def _init_weights(self):
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.res3a(x)
        x = self.res3b(x)
        x = self.res3c(x)
        x = self.res3d(x)

        x = self.res4a(x)
        x = self.res4b(x)
        x = self.res4c(x)
        x = self.res4d(x)
        x = self.res4e(x)
        x = self.res4f(x)

        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.avepool(x)
        x = self.fc(x)

        return x