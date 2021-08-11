"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import data
import torch
import torch.nn.functional as F

from util import html
from collections import OrderedDict
from options.test_options import TestOptions
from models.wavelet_model import WaveletModel
from util.visualizer import Visualizer
from pytorch_wavelets import DWTForward, DWTInverse

opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)
model = WaveletModel(opt)
model.eval()
xfm = DWTForward(J=opt.wavelet_decomp_level, mode='zero', wave='haar')
ifm = DWTInverse(mode='zero', wave='haar')
visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')
    Yl, Yh = xfm(data_i['image'][:,:3].cpu())
    masks = data_i['mask'].cpu()
    comp_images = data_i['masked_img'][:,:3].cpu() * (1-masks) + generated[-1].cpu() * masks
    img_path = data_i['img_name']
    for b in range(generated[-1].shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('masked_image', data_i['masked_img'][b,:3].cpu()),
                               ('synthesized_image', generated[-1][b].cpu()),
                               ('comp_image', comp_images[b].cpu()),
                               ('ground_truth', data_i['image'][b].cpu())])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
