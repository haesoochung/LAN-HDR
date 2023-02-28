# LAN-HDR
Pytorch implementation of our paper: LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction 

## Environments
- Ubuntu 18.04
- Pytorch 1.12.1
- CUDA 11.3 & cuDNN 8.3.2
- Python 3.9.6

## Preparation
#### Environment setting
Type the following command to build the environment after dowloading 'lan-hdr.yml'.

```
conda env create -f lan-hdr.yml
```
#### Data preparation
1. Download the [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset/) for training and testing.
2. Locate the downloaded datasets to "data" folder.

## Training

##### 2 Exposures
```
python main.py --affix 2E --nexps 2 --nframes 5 --gpu 0 --train
```
##### 3 Exposures
```
python main.py --affix 3E --nexps 3 --nframes 7 --gpu 0 --train
```

##### For DistributedDataParallel training
Use the following commands.
```
CUDA_VISIBLE_DEVICES=#,# python -m torch.distributed.launch --nnode=1 --nproc_per_node=2 --use_env main.py --affix 2E --nexps 2 --nframes 5 --gpu 0 --train
CUDA_VISIBLE_DEVICES=#,# python -m torch.distributed.launch --nnode=1 --nproc_per_node=2 --use_env main.py --affix 3E --nexps 3 --nframes 7 --gpu 0 --train
```


## Test
Specify the test dataset and its location with argument "--benchmark" and "--data_dir", respectively.

(Cinematic Video dataset: synthetic_dataset / DeepHDRVideo dataset: dynamic_dataset & static_dataset / HDRVideo dataset: tog13_online_align_dataset)

##### 2 Exposures
```
python main.py --affix 2E_test --nexps 2 --nframes 5 --benchmark synthetic_dataset --gpu 0 --data_dir ./data/HDR_Synthetic_Test_Dataset
python main.py --affix 2E_test --nexps 2 --nframes 5 --benchmark dynamic_dataset --gpu 0 --data_dir ./data/dynamic_RGB_data_2exp_release
python main.py --affix 2E_test --nexps 2 --nframes 5 --benchmark static_dataset --gpu 0 --data_dir ./data/static_RGB_data_2exp_rand_motion_release
python main.py --affix 2E_test --nexps 2 --nframes 5 --benchmark tog13_online_align_dataset --gpu 0 --data_dir ./data/TOG13_Dynamic_Dataset
```
##### 3 Exposures
```
python main.py --affix 3E_test --nexps 3 --nframes 7 --benchmark synthetic_dataset --gpu 0 --data_dir ./data/HDR_Synthetic_Test_Dataset
python main.py --affix 3E_test --nexps 3 --nframes 7 --benchmark dynamic_dataset --gpu 0 --data_dir ./data/dynamic_RGB_data_3exp_release
python main.py --affix 3E_test --nexps 3 --nframes 7 --benchmark static_dataset --gpu 0 --data_dir ./data/static_RGB_data_3exp_rand_motion_release
python main.py --affix 3E_test --nexps 3 --nframes 7 --benchmark tog13_online_align_dataset --gpu 0 --data_dir ./data/TOG13_Dynamic_Dataset
```
