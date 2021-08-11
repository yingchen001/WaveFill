import cv2
import os
import sys
import math
import time
import json
import glob
import torch
import argparse
import urllib.request
from PIL import Image, ImageFilter
from numpy import random
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from util.fid_score import calculate_activation_statistics, calculate_frechet_distance
from util.inception import InceptionV3

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-p', '--path', required=True, type=str)
args = parser.parse_args()

dims = 2048
batch_size = 4
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = InceptionV3([block_idx]).to(cuda)

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / (np.sum(img_true + img_test) + 1e-6)

def ssim(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        error += compare_ssim(frames1[i], frames2[i], multichannel=True, win_size=51)
    return error/len(frames1)

def psnr(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        error += compare_psnr(frames1[i], frames2[i])
    return error/len(frames1)

def mae(frames1, frames2):
    error = 0
    for i in range(len(frames1)):
        error += compare_mae(frames1[i], frames2[i])
    return error/len(frames1)

def main():
    real_names = list(glob.glob('{}/ground_truth/*.png'.format(args.path)))
    fake_names = list(glob.glob('{}/comp_image/*.png'.format(args.path)))
    real_names.sort()
    fake_names.sort()
    # metrics prepare for image assesments
    metrics = {'mae':mae, 'psnr':psnr, 'ssim':ssim}
    # infer through videos
    real_images = []
    fake_images = []
    evaluation_scores = {key: 0 for key,val in metrics.items()}
    for rname, fname in zip(real_names, fake_names):
        rimg = Image.open(rname)
        fimg = Image.open(fname)
        real_images.append(np.array(rimg))
        fake_images.append(np.array(fimg))
    # calculating image quality assessments
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images)
    message_full = 'Whole Image Metrics: '
    message_full += ' '.join(['{}: {:.4f},'.format(key, val) for key,val in evaluation_scores.items()])
    
    # calculate fid statistics for real images 
    real_images = np.array(real_images).astype(np.float32)/255.0
    real_images = real_images.transpose((0, 3, 1, 2))
    real_m, real_s = calculate_activation_statistics(real_images, model, batch_size, dims, cuda=torch.cuda.is_available())
    
    # calculate fid statistics for fake images
    fake_images = np.array(fake_images).astype(np.float32)/255.0
    fake_images = fake_images.transpose((0, 3, 1, 2))
    fake_m, fake_s = calculate_activation_statistics(fake_images, model, batch_size, dims, cuda=torch.cuda.is_available())

    fid_value = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)
    message_full += 'FID: {}'.format(round(fid_value, 4))
    print(message_full)
    print('Finish evaluation from {}'.format(args.resume))

if __name__ == '__main__':
    main()

            
