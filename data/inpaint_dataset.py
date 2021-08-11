"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from scipy.misc import imread, imresize
from random import shuffle
import util.util as util
import numpy as np
import random
import torch
import os
import math

class InpaintDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser = InpaintDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataroot='./dataset/psv/')
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(input_nc=3)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        root_dir = os.path.dirname(root)
        image_paths = [i for i in np.genfromtxt(os.path.join(root, opt.dataset_name, opt.phase+'.flist'), dtype=np.str, encoding='utf-8')]
        if opt.mask_type >= 3:
            mask_paths = [os.path.join(root_dir, 'pconv_masks/testing_mask_dataset/{}.png'.format(str(i).zfill(5))) 
                            for i in range(2000, 12000)]
            if opt.pconv_level > 0:
                mask_paths = [os.path.join(root_dir, 'pconv_masks/testing_mask_dataset/{}.png'.format(str(i).zfill(5))) 
                                for i in range(opt.pconv_level*2000, (opt.pconv_level+1)*2000)]
            mask_paths= mask_paths*(max(1, math.ceil(len(image_paths)/len(mask_paths))))
        else:
            mask_paths = ['0']*len(image_paths)
        return image_paths, mask_paths

    def initialize(self, opt):
        self.opt = opt

        image_paths, mask_paths = self.get_paths(opt)

        util.natural_sort(mask_paths)
        util.natural_sort(image_paths)

        self.mask_paths = mask_paths
        self.image_paths = image_paths
        if opt.phase == 'train':
            self.image_paths = self.image_paths
            shuffle(self.image_paths)
        self.mask_paths = self.mask_paths[:opt.max_dataset_size]
        self.image_paths = self.image_paths[:opt.max_dataset_size]
        if opt.debug:
            self.image_paths = self.image_paths[:100]
        size = len(self.image_paths)
        self.dataset_size = size
        self.h, self.w = opt.load_size, opt.load_size

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext
    
    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.image_paths[index])
            item = self.load_item(0)
        return item

    def load_item(self, index):
        
        img_path = self.image_paths[index]
        img_name = os.path.basename(self.image_paths[index])
        img = imread(img_path)
        # center crop and resize to target size
        img = self.resize(img, self.h, self.h)
        img = Image.fromarray(img).convert('RGB')
        mask  = self.load_mask(index)
        mask = Image.fromarray(mask)
        # augment 
        if self.opt.phase == 'train': 
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
            mask = transforms.RandomHorizontalFlip()(mask)
            # mask = mask.rotate(random.randint(0,45), expand=True)
            mask = mask.filter(ImageFilter.MaxFilter(3))
            
        img = img.resize((self.w, self.h), Image.BICUBIC)
        mask = mask.resize((self.w, self.h), Image.NEAREST)
        img_tensor = F.to_tensor(img)*2-1.
        mask_tensor = F.to_tensor(mask)
        masked_img = (img_tensor * (1-mask_tensor)) + mask_tensor
        masked_img = torch.cat([masked_img, mask_tensor], 0)

        input_dict = {'image': img_tensor,
                      'mask': mask_tensor,
                      'img_name': img_name,
                      'masked_img': masked_img
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def load_mask(self, index):
        mask_type = self.opt.mask_type

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + random stroke
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            m = np.zeros((self.h, self.w)).astype(np.uint8)
            if self.opt.phase == 'train':
                t, l = random.randint(0, self.h//2), random.randint(0, self.w//2)
                m[t:t+self.h//2, l:l+self.w//2] = 255
            else:
                m[self.h//4:self.h*3//4, self.w//4:self.w*3//4] = 255
            return m

        # generate random mask
        if mask_type == 2:
            mask = util.generate_stroke_mask([self.h, self.w])
            mask = self.resize(mask, self.h, self.w)
            return (mask>0).astype(np.uint8)* 255

        # external
        if mask_type == 3:
            m_index = random.randint(0, len(self.mask_paths)-1) if self.opt.phase == 'train' else index
            mask_path = self.mask_paths[m_index]
            mask = imread(mask_path)
            mask = self.resize(mask, self.h, self.w)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask
    
    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = imresize(img, [height, width])
        return img

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size