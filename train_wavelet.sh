python train_wavelet.py --name xxx_wavelet_fran --dataset_mode inpaint --dataroot ../../dataset/flist \
            --dataset_name psv --mask_type 4 --pconv_level 0 --niter 40 --niter_decay 40 \
            --netG WaveletInpaintLv2GCFRAN --batchSize 10 --n_layers_D 4 --ngf 64 --wavelet_decomp_level 2 \
            --input_nc 4 --output_nc 3 --highfreq_nc 12 --gan_mode hinge --use_attention \
            --lambda_perceptual 0.001 --lambda_vgg 10 --lambda_style 0 --vgg_normal_correct  \
            --lambda_dwt_l 2 --lambda_gan_h 1  --lambda_feat_h 5 \
            --gpu_ids 0 --continue_train \
