"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, FRAN
from models.networks.normalization import get_nonspade_norm_layer


class Attention(nn.Module):
    def __init__(self, ch, use_sn, with_attn=False):
        super(Attention, self).__init__()
        # Channel multiplier
        self.with_attn = with_attn 
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        out = self.gamma * o + x
        if self.with_attn:
            return out, beta
        else:
            return out

# FRAN ResBlk
class GC_FRANResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, highfreq_nc=None, groups=1, ch_rate=4, size_rate=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        if highfreq_nc is None:
            highfreq_nc = opt.highfreq_nc

        # create conv layers
        self.conv_0 = GatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, activation=None)
        self.conv_1 = GatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, activation=None)
        if self.learned_shortcut:
            self.conv_s = GatedConv2dWithActivation(fin, fout, kernel_size=1, bias=False, activation=None)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = GatedConv2dWithActivation(fin, fmiddle, kernel_size=3, padding=1, norm_layer=spectral_norm, activation=None)
            self.conv_1 = GatedConv2dWithActivation(fmiddle, fout, kernel_size=3, padding=1, norm_layer=spectral_norm, activation=None)
            if self.learned_shortcut:
                self.conv_s = GatedConv2dWithActivation(fin, fout, kernel_size=1, bias=False, norm_layer=spectral_norm, activation=None)


        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = FRAN(spade_config_str, fin, highfreq_nc, ch_rate, size_rate)
        self.norm_1 = FRAN(spade_config_str, fmiddle, highfreq_nc, ch_rate, size_rate)
        if self.learned_shortcut:
            self.norm_s = FRAN(spade_config_str, fin, highfreq_nc, ch_rate, size_rate)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), dilation=1, kernel_size=3):
        super().__init__()

        pw_d = dilation * (kernel_size - 1) // 2
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw_d),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, dilation=dilation)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size, dilation=1))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out

# ResNet block with Gated Conv
class ResnetBlock_GC(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), dilation=1, kernel_size=3, groups=1):
        super().__init__()

        pw_d = dilation * (kernel_size - 1) // 2
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw_d),
            GatedConv2dWithActivation(dim, dim, kernel_size=kernel_size, dilation=dilation,
                                    groups=groups, norm_layer=norm_layer, activation=activation),
            nn.ReflectionPad2d(pw),
            GatedConv2dWithActivation(dim, dim, kernel_size=kernel_size, dilation=1, 
                                    groups=groups, norm_layer=norm_layer, activation=None)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                bias=True, norm_layer=None, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = activation
        if norm_layer is not None:
            self.conv2d = norm_layer(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.mask_conv2d = norm_layer(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        else:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        
        return x

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
