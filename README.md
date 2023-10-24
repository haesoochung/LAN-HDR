# LAN-HDR

This repository contains the official Pytorch implementation of the following paper:

> **LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction**<br>
> Haesoo Chung and Nam Ik Cho<br>
> https://arxiv.org/abs/2308.11116
>
> **Abstract:** *As demands for high-quality videos continue to rise, high-resolution and high-dynamic range (HDR) imaging techniques are drawing attention. To generate an HDR video from low dynamic range (LDR) images, one of the critical steps is the motion compensation between LDR frames, for which most existing works employed the optical flow algorithm. However, these methods suffer from flow estimation errors when saturation or complicated motions exist. In this paper, we propose an end-to-end HDR video composition framework, which aligns LDR frames in the feature space and then merges aligned features into an HDR frame, without relying on pixel-domain optical flow. Specifically, we propose a luminance-based alignment network for HDR (LAN-HDR) consisting of an alignment module and a hallucination module. The alignment module aligns a frame to the adjacent reference by evaluating luminance-based attention, excluding color information. The hallucination module generates sharp details, especially for washed-out areas due to saturation. The aligned and hallucinated features are then blended adaptively to complement each other. Finally, we merge the features to generate a final HDR frame. In training, we adopt a temporal loss, in addition to frame reconstruction losses, to enhance temporal consistency and thus reduce flickering. Extensive experiments demonstrate that our method performs better or comparable to state-of-the-art methods on several benchmarks.*


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
1. Download the [DeepHDRVideo-Dataset](https://github.com/guanyingc/DeepHDRVideo-Dataset/) for training and testing. (All the data can be downloaded from the provided Onedrive link.)
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

- Cinematic Video dataset: synthetic_dataset
- DeepHDRVideo dataset: dynamic_dataset & static_dataset
- HDRVideo dataset: tog13_online_align_dataset

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

## Citation
```
@inproceedings{chung2023lan,
  title={LAN-HDR: Luminance-based Alignment Network for High Dynamic Range Video Reconstruction},
  author={Chung, Haesoo and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12760--12769},
  year={2023}
}
```
