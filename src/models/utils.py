#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Sun Apr 21 2019 
@Time    : 17:47:53
@File    : utils.py
@Author  : alpha
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Scale(nn.Module):

    def __init__(self, num_features):
        super(Scale, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.ones(self.num_features))
        self.bias = Parameter(torch.zeros(self.num_features))

    def forward(self, x):
        reshaped_size = (-1,) + (1,) * (len(x.shape) - 2)
        return x * self.weight.reshape(*reshaped_size) + self.bias.reshape(*reshaped_size)


## TODO: design a template layer to easily inherit common layers to normed ones
class _Normed:
    def __init__(self):
        pass


class NormedConv2D(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(NormedConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, False)

        self.register_parameter('_normed_weight', None)
        self.register_forward_pre_hook(self._get_normed_weight)

        self._scale_weight = Parameter(torch.full((self.out_channels,), 0.1))
        self._scale_bias = Parameter(torch.zeros(self.out_channels))

        self.register_parameter('fused_weight', None)
        self.register_parameter('fused_bias', None)
        self.register_forward_hook(self._get_fused_weights)

    @staticmethod
    def _get_normed_weight(self, *_):
        if self.training:
            weight = self.weight
            reshaped_size = (-1,) + (1,) * (len(weight.shape) - 1)
            weight_mean = weight.view(weight.size(0), -1).mean(1).view(*reshaped_size)
            weight = weight - weight_mean
            weight_std = weight.view(weight.size(0), -1).std(1).view(*reshaped_size)
            weight_std = torch.clamp(weight_std, 1e-3)
            self._normed_weight = Parameter(weight / weight_std.expand_as(weight))

    @staticmethod
    def _get_fused_weights(self, *_):
        if self.training:
            reshaped_size = (-1,) + (1,) * (len(self._normed_weight.shape) - 1)
            self.fused_weight = Parameter(self._scale_weight.view(*reshaped_size) * self._normed_weight)
            self.fused_bias = Parameter(self._scale_bias)

    def forward(self, x):
        if self.training:
            x = F.conv2d(x, self._normed_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = x * self._scale_weight.reshape(-1, 1, 1) + self._scale_bias.reshape(-1, 1, 1)
        else:
            x = F.conv2d(x, self.fused_weight, self.fused_bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class NormedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__(in_features, out_features, False)

        self.register_parameter('_normed_weight', None)
        self.register_forward_pre_hook(self._get_normed_weight)

        self._scale_weight = Parameter(torch.ones(self.out_features))
        self._scale_bias = Parameter(torch.zeros(self.out_features))

        self.register_parameter('fused_weight', None)
        self.register_parameter('fused_bias', None)
        self.register_forward_hook(self._get_fused_weights)

    @staticmethod
    def _get_normed_weight(self, *_):
        if self.training:
            weight = self.weight
            reshaped_size = (-1,) + (1,) * (len(weight.shape) - 1)
            weight_mean = weight.view(weight.size(0), -1).mean(1).view(*reshaped_size)
            weight = weight - weight_mean
            weight_std = weight.view(weight.size(0), -1).std(1).view(*reshaped_size)
            weight_std = torch.clamp(weight_std, 1e-3)
            self._normed_weight = Parameter(weight / weight_std.expand_as(weight))

    @staticmethod
    def _get_fused_weights(self, *_):
        if self.training:
            reshaped_size = (-1,) + (1,) * (len(self._normed_weight.shape) - 1)
            self.fused_weight = Parameter(self._scale_weight.view(*reshaped_size) * self._normed_weight)
            self.fused_bias = Parameter(self._scale_bias)

    def forward(self, x):
        if self.training:
            x = F.linear(x, self._normed_weight, self.bias)
            x = x * self._scale_weight + self._scale_bias
        else:
            x = F.conv2d(x, self.fused_weight, self.fused_bias)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)