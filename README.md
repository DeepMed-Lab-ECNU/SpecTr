# SpecTr: Spectral Transformer for Microscopic Hyperspectral Pathology Image Segmentation (TCSVT 2023)

Official Code for "SpecTr: Spectral Transformer for Microscopic Hyperspectral Pathology Image Segmentation"
by Boxiang Yun, Baiying Lei, Jieneng Chen, Huiyu Wang, Song Qiu, Wei Shen, Qingli Li, Yan Wang*

## Introduction
(TCSVT 2023) Official code for "[SpecTr: Spectral Transformer for Microscopic Hyperspectral Pathology Image Segmentation](https://ieeexplore.ieee.org/abstract/document/10288474)".
![SpecTr](https://github.com/DeepMed-Lab-ECNU/SpecTr/assets/36001411/38346e9e-bf97-441f-a099-b3b4f729b584)


## Requirements
This repository is based on PyTorch 1.10, CUDA 11.１, Python 3.９.７, and segmentation-models-pytorch 0.3.3. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Usage
We provide `code`, `dataset`, and `model` for the MDC dataset.

The official dataset can be found at [MDC](http://bio-hsi.ecnu.edu.cn/). However, due to its size, we also provide preprocessed [data](https://www.kaggle.com/datasets/hfutybx/mhsi-choledoch-dataset-preprocessed-dataset) (including denoising and resizing operations) for reproducing our paper experiments." 

Download the dataset and move to the dataset fold.

To train a model,
```
CUDA_VISIBLE_DEVICES=0 python train_main.py -r ./dataset/MDC -sn 60 -cut 192 -e 75
```

To test a model,
```
CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python evaluate.py -r ./dataset/MDC -sn 60 -cut 192 -name SpecTr_XXXX
```

## Acknowledgements
Some modules in our code were inspired by [vit-pytorch](https://github.com/lucidrains/vit-pytorch) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). We appreciate the effort of these authors to provide open-source code for the community. Hope our work can also contribute to related research.

## Questions
If you encounter any issues accessing the dataset, such as unable to sign in [MDC](http://bio-hsi.ecnu.edu.cn/), please contact me at 'boxiangyun@gmail.com'
