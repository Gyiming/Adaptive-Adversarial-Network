# Input-Dependent-ADV
This repo contains the artifact of MORPHADNET: Dynamic and Adaptive Adversarial Training, which is currently under review of CVPR 2022.

The goal is to condition an adversarial network based on inference-time detection of input characteristics.

## Working environment
We have tested our code on a system with Red Hat 4.8.5-39; the machine we run this code is Intel(R) Xeon(R) Silver 4110 with 96115 MB memory in total. The machine has two NVIDIA GeForce 2080Ti GPU with CUDA version 9.0.176. 

## Install
```bash
cd <path-to-project>
conda env create -f environment.yml
source activate morphadnet
```

## Download pretrained weights

All the pre-trained weights will be available online.

## Pre-process datasets

### CIFAR-10 and CIFAR-100

The two datasets will be automatically downloaded if not exist.

## Baselines
```bash
python PGDAT_1.py
```

## Training
```bash
python 2BN_dyn.py
```
 
