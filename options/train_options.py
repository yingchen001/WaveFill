"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=5000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        # parser.add_argument('--eval', action='store_true', help='whether run evaluation')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--G_steps_per_D', type=int, default=1, help='number of generator iterations per discriminator iterations.')
        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--no_perc_loss', action='store_true', help='if specified, do *not* use VGG-related losses')
        parser.add_argument('--vgg_normal_correct', action='store_true', help='if true, correct vgg normalization')
        parser.add_argument('--lambda_vgg', type=float, default=5.0, help='weight for vgg loss')
        parser.add_argument('--lambda_perceptual', type=float, default=0.001, help='weight for perceptual loss')
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for style loss')
        parser.add_argument('--lambda_hole', type=float, default=6.0, help='weight for l1 loss of hole')
        parser.add_argument('--lambda_dwt_l', type=float, default=1.0, help='weight for wavelet low freq loss')
        parser.add_argument('--lambda_gan_h', type=float, default=1.0, help='weight for wavelet high freq loss')
        parser.add_argument('--lambda_feat_h', type=float, default=1.0, help='weight for high freq feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--lambda_kld', type=float, default=0.05)
        self.isTrain = True
        return parser
