"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import torch
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import Attention
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc=3):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, mask):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            # input = self.downsample(input)
            # local discriminator: center crop for regular mask; random crop for irregular mask
            b, c, h, w = input.shape
            if self.opt.mask_type == 1:
                h_local = h // 2
                w_local = w // 2
                crops = []
                for i in range(b):
                    crop_x, crop_y = torch.nonzero(mask[0,0]).min(dim=0).values
                    crops.append(input[i, :, crop_x:crop_x+h_local, crop_y:crop_y+w_local])
                input = torch.stack(crops, 0)
            else:
                h_local = h // 2
                w_local = w // 2
                crop_x = random.randint(0, h-h_local-1)
                crop_y = random.randint(0, w-w_local-1)
                input = input[:, :, crop_x:crop_x + h_local, crop_y:crop_y + w_local]

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int((kw - 1.0) / 2)
        # padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        # input_nc = self.compute_D_input_nc(opt)
        # input_nc = opt.output_nc
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            if opt.use_attention and n == opt.n_layers_D - 1:
                self.attn = Attention(nf_prev, 'spectral' in opt.norm_D)
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.input_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for name, submodel in self.named_children():
            if 'model' not in name:
                continue
            if name == 'model{}'.format(self.opt.n_layers_D - 1):
                if self.opt.use_attention:
                    x = self.attn(results[-1])
                else:
                    x = results[-1]
            else:
                x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)
        # for submodel in self.children():
        #     intermediate_output = submodel(results[-1])
        #     results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
