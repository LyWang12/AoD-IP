# Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs

Code release for "Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs" (CVPR 2026)

## Paper

<div align=center><img src="https://github.com/LyWang12/AoD-IP/blob/main/Figure/1.png" width="100%"></div>


[Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs](https://arxiv.org/abs/2603.04896) 
(CVPR 2026)

We proposed a novel Authorize-on-Demand (AoD) IP protection framework for vision-language models, which enables flexible, user-controlled IP protection through a lightweight on-demand authorization module and a dual-path inference mechanism for robust task-specific performance and illegal-domain detection.

<div align=center><img src="https://github.com/LyWang12/AoD-IP/blob/main/Figure/2.png" width="100%"></div>

## Prerequisites
The code is implemented with **CUDA 11.4**, **Python 3.8.5** and **Pytorch 1.8.0**.

## Datasets

### Office-31
Office-31 dataset can be found [here](https://opendatalab.com/OpenDataLab/Office-31).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset).

### Mini-DomainNet
Mini-DomainNet dataset can be found [here](https://github.com/KaiyangZhou/Dassl.pytorch).


## Running the code

Target-Specified IP-CLIP
```
python Target_Specified/train_vit_b16.py
```

Applicability Authorization by IP-CLIP
```
python Authorization/train_vit_b16.py
```


## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com



