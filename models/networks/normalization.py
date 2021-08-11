"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        elif subnorm_type == 'RN_L':
            norm_layer = RN_L(get_out_channel(layer))
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |input_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, input_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'position':
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class AttentionNorm(nn.Module):
    def __init__(self, ch, use_sn, rate=1, temp=5, with_attn=False):
        super(AttentionNorm, self).__init__()
        # Channel multiplier
        self.with_attn = with_attn 
        self.temp = temp
        self.ch = ch
        self.down0 = nn.Conv2d(self.ch, self.ch, kernel_size=rate, padding=rate//4, stride=rate)
        self.down1 = nn.Conv2d(self.ch, self.ch, kernel_size=rate, padding=rate//4, stride=rate)
        self.theta = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        self.up = nn.ConvTranspose2d(self.ch, self.ch, kernel_size=rate, padding=0, stride=rate)
        if use_sn:
            self.down0 = spectral_norm(self.down0)
            self.down1 = spectral_norm(self.down1)
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
            self.up = spectral_norm(self.up)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input_x, input_y):
        # Apply convs
        x = self.down0(input_x)
        y = self.down1(input_y)
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(y), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 2, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(self.temp * torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        out = self.up(o) + self.gamma * input_y
        if self.with_attn:
            return out, beta
        else:
            return out

class FRAN(nn.Module):
    def __init__(self, config_text, norm_nc, input_nc, ch_rate=4, size_rate=1):
        super().__init__()

        assert config_text.startswith('fran')
        parsed = re.search('fran(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if param_free_norm_type == 'position':
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False, track_running_stats=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = norm_nc//ch_rate

        pw = ks // 2
        self.mlp_shared0 = nn.Sequential(
            nn.Conv2d(norm_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_shared1 = nn.Sequential(
            nn.Conv2d(input_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.an = AttentionNorm(nhidden, True, size_rate)
                                
    def forward(self, x, segmap):
        
        normalized = self.param_free_norm(x)    
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')


        # gamma and beta
        actv0 = self.mlp_shared0(x)
        actv1 = self.mlp_shared1(segmap)

        actv = self.an(actv0, actv1)  
        actv = self.param_free_norm(actv)
        gamma = self.mlp_gamma(actv) 
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out