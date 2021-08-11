from PIL import Image
import os
import numpy as np

bs_dir = '/home/fangneng.zfn/datasets/Alibaba/spade/2020-06-24/'
im_dir = bs_dir + 'train_input_/'
sv_dir = bs_dir + 'train_input/'

nms = os.listdir(im_dir)
# nms = nms[519:]
i = 0
for nm in nms:
    # print (nm)
    im_path = im_dir + nm
    im = Image.open(im_path)
    tmp = nm.split('_uv2front')[0] + '.png'
    sv_path = sv_dir + tmp
    im.save(sv_path)
    i += 1
    print (i)
