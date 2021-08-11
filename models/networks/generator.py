"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock_GC as ResnetBlock_GC
from models.networks.architecture import GC_FRANResnetBlock as GC_FRANResnetBlock
from models.networks.architecture import GatedConv2dWithActivation as GatedConv2d
from models.networks.architecture import Attention as Attention


class WaveletInpaintLv2GCFRANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralfranposition3x3')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.use_cuda = (len(self.opt.gpu_ids) > 0) and (-1 not in self.opt.gpu_ids)
        nf = opt.ngf
        final_nc = nf
        activation = nn.ReLU(False)
        norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
        self.fc = GatedConv2d(self.opt.input_nc, 4 * nf, 3, padding=1, norm_layer=norm_layer, activation=activation)
        self.encoder = nn.Sequential(nn.ReflectionPad2d(2),
                    GatedConv2d(self.opt.input_nc, nf, kernel_size=5, padding=0, norm_layer=norm_layer, activation=activation),
                    GatedConv2d(nf, 2*nf, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation=activation),
                    GatedConv2d(2*nf, 4*nf, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, activation=activation))
        self.fuse_conv = GatedConv2d(8 * nf, 4 * nf, 3, padding=1, norm_layer=norm_layer, activation=activation)
        self.xfm = DWTForward(J=opt.wavelet_decomp_level, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')
        res_blocks = []
        for i in range(opt.resnet_n_blocks):
            dilation = 2**(i-2) if i <= 6 and i>=3 else 1
            res_blocks += [ResnetBlock_GC(4 * nf,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  dilation=dilation,
                                  kernel_size=opt.resnet_kernel_size)]
        self.res_blocks = nn.Sequential(*res_blocks)
        if self.opt.use_attention:
            self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)
        self.sp0 = GC_FRANResnetBlock(8 * nf, 4 * nf, opt, ch_rate=4, size_rate=1)
        self.sp1 = GC_FRANResnetBlock(12 * nf, 2 * nf, opt, ch_rate=4, size_rate=2)


        self.up = nn.Upsample(scale_factor=2)
        # branch 1:
        self.res_b1 = ResnetBlock_GC(4 * nf, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)
        self.conv_b1 = GatedConv2d(4 * nf, self.opt.output_nc, 3, padding=1, activation=None)

        # branch 2:
        self.res_b2 = ResnetBlock_GC(4 * nf, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)
        self.conv_b2 = GatedConv2d(4 * nf, self.opt.output_nc * 3, 3, padding=1, activation=None)

        # branch 3:
        self.res_b3 = ResnetBlock_GC(2 * nf, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)
        self.conv_b3 = GatedConv2d(2 * nf, self.opt.output_nc * 3, 3, padding=1, activation=None)


    def forward(self, input, z=None):
        image, mask = input[:,:3], input[:,3:4]
        b, c, h, w = input.shape
        Yl, Yh = self.xfm(input) # Yl.shape = [b, c, h//8, w//8]
        Yh1 = Yh[0].contiguous().view(b,c*3,h//2,w//2)
        Yh0 = Yh[1].contiguous().view(b,c*3,h//4,w//4)
        x = self.fc(Yl)
        x_enc = self.encoder(input)
        x = self.fuse_conv(torch.cat([x, x_enc], 1))
        x = self.res_blocks(x)
        if self.opt.use_attention:
            x = self.attn(x)
        # branch 1: recover Yl
        x_l = self.res_b1(x)
        out_l = self.conv_b1(x_l)
        x = torch.cat([x, x_l] ,dim=1)
        x = self.sp0(x, Yh0)

        # branch 2: recover Yh0
        x_h0 = self.res_b2(x)
        out_h0 = self.conv_b2(x_h0) 
        x = torch.cat([x, x_l, x_h0] ,dim=1)
        x = self.up(x)
        x = self.sp1(x, Yh1)

        # branch 3: recover Yh1
        x_h1 = self.res_b3(x)
        out_h1 = self.conv_b3(x_h1) 
        
        # high-frequency outputs
        out_h = [out_h1.contiguous().view(b,self.opt.output_nc,3,h//2,w//2), 
                out_h0.contiguous().view(b,self.opt.output_nc,3,h//4,w//4)] 
        out_pyramid = [self.ifm((out_l, out_h[1:]))] # pyramid outputs (eg. 64x64, 128x128)
        out_wavelet = self.ifm((out_l, out_h)) # final output

        return out_l, out_h, out_pyramid, out_wavelet
