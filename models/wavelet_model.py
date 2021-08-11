"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import numpy as np
import models.networks as networks
import util.util as util
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from thop import profile, clever_format
class WaveletModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD_h, self.netE = self.initialize_networks(opt)

        self.xfm = DWTForward(J=self.opt.wavelet_decomp_level, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')
        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionL1 = torch.nn.L1Loss()
            if not opt.no_perc_loss:
                if opt.vgg_normal_correct:
                    self.criterionVGGs = networks.VGGLosses_fix(opt.vgg_normal_correct)
                else:
                    self.criterionVGGs = networks.VGGLosses(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_h_params = [list(self.netD_h[i].parameters()) for i in range(self.opt.wavelet_decomp_level)]

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))

        optimizer_D_h = [torch.optim.Adam(D_h_params[i], lr=D_lr, betas=(beta1, beta2)) for i in range(self.opt.wavelet_decomp_level)]

        return optimizer_G, optimizer_D_h

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        for i in range(self.opt.wavelet_decomp_level):
            util.save_network(self.netD_h[i], 'D_h_{}'.format(i), epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)

        # No. Channels = opt.output_nc*3 for high-frequency bands
        netD_h = torch.nn.ModuleList(networks.define_D(opt, opt.output_nc*3) if opt.isTrain  else None 
                    for i in range(self.opt.wavelet_decomp_level))
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            print('Generator Loaded')
            if opt.isTrain:
                try:
                    netD_h = torch.nn.ModuleList(util.load_network(netD_h[i], 'D_h_{}'.format(i), opt.which_epoch, opt)
                                                for i in range(self.opt.wavelet_decomp_level))
                except:
                    print('unable to load D_h')
                print('Discriminator Loaded')
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
                print('Encoder Loaded')

        return netG, netD_h, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['image'] = data['image'].cuda()
            data['masked_img'] = data['masked_img'].cuda()
        return data['masked_img'], data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}
        masks = input_semantics[:,3:4]
        fake_images, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)
        pred_l, pred_h, pred_wavelet = fake_images[0], fake_images[1], fake_images[-1]
        
        # Low frequency losses
        gt_l, gt_h = self.xfm(real_image) 
        scale_factor = 1/(2**self.opt.wavelet_decomp_level)
        masks_l = F.interpolate(masks, scale_factor=scale_factor)

        hole_loss_l = self.criterionL1(pred_l * masks_l, gt_l * masks_l) / torch.mean(masks_l) 
        valid_loss_l = self.criterionL1(pred_l * (1-masks_l), gt_l * (1-masks_l)) / torch.mean(1-masks_l) 
        l1_loss_l = scale_factor * self.opt.lambda_dwt_l * (hole_loss_l * self.opt.lambda_hole + valid_loss_l)
        G_losses['L1_L'] = l1_loss_l

        # High frequency losses
        
        for lv in range(len(gt_h)):
            pred_fake_h, pred_real_h = self.discriminate_highfreq(input_semantics, pred_h[lv], gt_h[lv], self.opt.wavelet_decomp_level-lv-1)
            gan_h_loss = self.opt.lambda_gan_h * self.criterionGAN(pred_fake_h, True, for_discriminator=False) 
            G_losses['GAN_H{}'.format(self.opt.wavelet_decomp_level-lv-1)] = gan_h_loss

            # Feature matching loszs
            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake_h)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake_h[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionL1(
                            pred_fake_h[i][j], pred_real_h[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat_h / num_D
                G_losses['GAN_Feat{}'.format(self.opt.wavelet_decomp_level-lv-1)] = GAN_Feat_loss


        # Whole image losses
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        if not self.opt.no_perc_loss:
            vgg_loss, style_loss, perc_loss = self.criterionVGGs(fake_images[-1], real_image) 
            G_losses['perc'] = self.opt.lambda_perceptual * perc_loss
            G_losses['VGG'] = self.opt.lambda_vgg * vgg_loss
            G_losses['Style'] = self.opt.lambda_style * style_loss

        return G_losses, fake_images

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_images, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_images[-1].detach()
            fake_image.requires_grad_()
            fake_lowfreq = fake_images[0].detach()
            fake_lowfreq.requires_grad_()
            fake_highfreqs = [x.detach().requires_grad_() for x in fake_images[1]]

        real_lowfreq, real_highfreqs = self.xfm(real_image)
        for i in range(len(real_highfreqs)):
            fake_highfreqs[i].requires_grad_()
            pred_fake_h, pred_real_h = self.discriminate_highfreq(input_semantics, fake_highfreqs[i], real_highfreqs[i], len(real_highfreqs)-i-1)

            D_losses['D_fake_h{}'.format(len(real_highfreqs)-i-1)] = self.criterionGAN(pred_fake_h, False,
                                                                    for_discriminator=True)
            D_losses['D_real_h{}'.format(len(real_highfreqs)-i-1)] = self.criterionGAN(pred_real_h, True,
                                                                    for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_images = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_images, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate_highfreq(self, input_semantics, fake_image, real_image, level):
        assert fake_image.shape == real_image.shape

        b, c, j, h, w = fake_image.shape
        fake_image = fake_image.contiguous().view(b,c*j,h,w)
        real_image = real_image.contiguous().view(b,c*j,h,w)
        mask = F.interpolate(input_semantics[:,3:4], fake_image.shape[2:])
        mask = torch.cat([mask, mask], dim=0)
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.netD_h[level](fake_and_real, mask)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
