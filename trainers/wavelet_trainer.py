"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.wavelet_model import WaveletModel
from torch import autograd
import torch

class WaveletTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.wavelet_model = WaveletModel(opt)
        print(opt.gpu_ids)
        if len(opt.gpu_ids) > 0:
            self.wavelet_model.to(f'cuda:{opt.gpu_ids[0]}')
            self.wavelet_model = DataParallelWithCallback(self.wavelet_model,
                                                          device_ids=opt.gpu_ids)
            self.wavelet_model_on_one_gpu = self.wavelet_model.module
        else:
            self.wavelet_model_on_one_gpu = self.wavelet_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D_h = self.wavelet_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
    
    def run_generator_one_step(self, data, print_gradient=False):
        with autograd.detect_anomaly():
            
            self.set_requires_grad(self.wavelet_model_on_one_gpu.netG, True)
            for i in range(self.opt.wavelet_decomp_level):
                self.set_requires_grad(self.wavelet_model_on_one_gpu.netD_h[i], False)
            
            self.optimizer_G.zero_grad()
            g_losses, generated = self.wavelet_model(data, mode='generator')

            g_loss = sum(g_losses.values()).mean()
            g_loss.backward()
            if print_gradient:
                for name, p in self.wavelet_model_on_one_gpu.netG.named_parameters():
                    if 'up' in name or 'conv_b' in name:
                        print('===========\ngradient:{}, {}'.format(name, p.grad.norm()))
            self.optimizer_G.step()
            self.g_losses = g_losses
            self.generated = generated

    def run_discriminator_one_step(self, data, print_gradient=False):
        with autograd.detect_anomaly():
            self.set_requires_grad(self.wavelet_model_on_one_gpu.netG, False)
            for i in range(self.opt.wavelet_decomp_level):
                self.set_requires_grad(self.wavelet_model_on_one_gpu.netD_h[i], True)

            d_losses = self.wavelet_model(data, mode='discriminator')
            for i in range(self.opt.wavelet_decomp_level):
                self.optimizer_D_h[i].zero_grad()
                d_loss_h = ((d_losses['D_fake_h{}'.format(i)]+d_losses['D_real_h{}'.format(i)])/2).mean()
                d_loss_h.backward()
                self.optimizer_D_h[i].step()

            self.d_losses = d_losses
        
    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.wavelet_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
