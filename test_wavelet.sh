python3 test_wavelet.py --name psv_wavelet_fran --dataset_mode inpaint --dataroot ../../dataset/flist \
            --dataset_name psv --netG WaveletInpaintLv2GCFRAN --mask_type 1 --pconv_level 0 --batchSize 1 \
            --input_nc 4 --output_nc 3 --highfreq_nc 12 --wavelet_decomp_level 2 --ngf 64 --use_attention