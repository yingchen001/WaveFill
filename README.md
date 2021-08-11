# WaveFill: A Wavelet-based Generation Network for Image Inpainting

## Installation
```bash
pip install -r requirements.txt
```
This code requires the pytorch_wavelets package.
```
$ git clone https://github.com/fbcotter/pytorch_wavelets
$ cd pytorch_wavelets
$ pip install .
```
## Dataset Preparation
Given the dataset, please prepare the images paths in a folder named by the dataset with the following folder strcuture.
```
    flist/dataset_name
        ├── train.flist    # Relative or absolute paths of training images
        ├── valid.flist    # Relative or absolute paths of validation images
        └── test.flist     # Relative or absolute paths of testing images
```
In this work, we use CelebA-HQ (Download availbale [here](https://github.com/switchablenorms/CelebAMask-HQ)), Places2 (Download availbale [here](http://places2.csail.mit.edu/download.html)), ParisStreet View (need author's permission to download)
## Testing with Pre-trained Models

1. Download pre-trained models: [Places2](https://drive.google.com/file/d/1GbkmyswAU47E4bZ-7AkA_YLZ8GvgtINj/view?usp=sharing) | [CelebA-HQ](https://drive.google.com/file/d/12nbh2nGA0VdBeVHXJx9trcr2fAImzUug/view?usp=sharing) | [Paris-StreetView](https://drive.google.com/file/d/1yM6oQgIBibsomgiYCLHNAC1dLhSnGJxx/view?usp=sharing)
2. Put the pre-trained model under the checkpoints folder, e.g.
```
    checkpoints
        ├── celebahq_wavefill
            ├── latest_net_G.pth 
```
3. Modify the bash file and run it.
```bash
# To specify dataset, name of pretain model and mask type in the bash file.
bash test_wavelet.sh
```

## Training New Models
**Pretrained VGG model** Download from [here](https://drive.google.com/file/d/1fp7DAiXdf0Ay-jANb8f0RHYLTRyjNv4m/view?usp=sharing), move it to `models/`. This model is used to calculate training loss.

New models can be trained with the following commands.

1. Prepare dataset. Use `--dataroot` option to locate the directory of file lists, e.g. `./flist`, and specify the dataset name to train with `--dataset_name` option. Identify the types and mask ratio using `--mask_type` and `--pconv_level` options. 

2. Train.
```bash
# To specify your own dataset or settings in the bash file.
bash train_wavelet.sh
```

There are many options you can specify. Please use `python train_wavelet.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

## Testing

Testing is similar to training new models.

```bash
python test_wavelet.py --name [name_of_experiment] --dataset_name [dataset_name] --dataroot [path_to_flist]
```

Use `--results_dir` to specify the output directory. `--how_many` will specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

## Acknowledgments
This code borrows heavily from [SPADE](https://github.com/NVlabs/SPADE), [CoCosNet](https://github.com/microsoft/CoCosNet), [PEN-Net](https://github.com/researchmm/PEN-Net-for-Inpainting) and [Edge-Connect](https://github.com/knazeri/edge-connect), we apprecite the authors for sharing their codes. We also thank Cotter for sharing the [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets/) code.