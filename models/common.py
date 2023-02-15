import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deformable_conv import ConvOffset2D


####################################################################################
# Single upsampling Base models
# scale: x2, x3, x4
####################################################################################
def default_conv(in_channels, out_channels, kernel_size, bias=True, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=(kernel_size//2), bias=bias)


class Upsampler(nn.Sequential):
    """
    使用pixelShuffle模块对特征图进行放大, 放大因子采用2,3,4,8
    """
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Deform_Upsampler(nn.Sequential):
    """
    使用pixelShuffle模块对特征图进行放大, 放大因子采用2,3,4,8
    """
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(ConvOffset2D(filters=n_feats))
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, bias, padding=1))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(ConvOffset2D(filters=n_feats))
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, bias, padding=1))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Deform_Upsampler, self).__init__(*m)


####################################################################################
# Attention Block
####################################################################################
# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Neuron Attention
class Neuron_Attention(nn.Module):
    def __init__(self, channel):
        super(Neuron_Attention, self).__init__()
        self.dw_conv = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.pw_conv = nn.Conv2d(channel, channel, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.dw_conv(x))
        y = self.pw_conv(y)
        y = self.sigmoid(y)
        return x * y


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


