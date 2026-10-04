# Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs

Code release for "Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs" (CVPR 2026)

## Paper

<div align=center><img src="https://github.com/LyWang12/AoD-IP/blob/main/Figure/1.png" width="100%"></div>


[Authorize-on-Demand: Dynamic Authorization with Legality-Aware Intellectual Property Protection for VLMs](https://arxiv.org/abs/2603.04896) 
(CVPR 2026)

We proposed a novel Authorize-on-Demand (AoD) IP protection framework for vision-language models, which enables flexible, user-controlled IP protection through a lightweight on-demand authorization module and a dual-path inference mechanism for robust task-specific performance and illegal-domain detection.

<div align=center><img src="https://github.com/LyWang12/AoD-IP/blob/main/Figure/2.png" width="100%"></div>

## Prerequisites
The code is tested with **Python 3.10**, **PyTorch 2.8.0** and **CUDA 12.8** (NVIDIA RTX PRO 6000, 96 GB).
PyTorch >= 1.13 is required (`torch.load(..., weights_only=False)`); older GPUs work with any CUDA build that matches your driver.

```
conda create -n aod python=3.10 -y
conda activate aod
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The CLIP ViT-B/16 weights are downloaded automatically to `Target_Specified/assets/` on first run (about 335 MB).

## Datasets

Put the datasets next to the code, i.e. `../Datasets/` relative to `Target_Specified/`, with the split files used in the paper:

```
Datasets/
├── office_home/   {art, clipart, product, real_world}/  split_random/{domain}_{train,test}.txt
├── domainnet/     {clipart, painting, real, sketch}/     splits_mini_random/{domain}_{train,test}.txt
└── office31/      {amazon, dslr, webcam}/                split_random/{domain}_{train,test}.txt
```

### Office-31
Office-31 dataset can be found [here](https://opendatalab.com/OpenDataLab/Office-31).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset).

### Mini-DomainNet
Mini-DomainNet dataset can be found [here](https://github.com/KaiyangZhou/Dassl.pytorch).


## Running the code

Target-Specified IP-CLIP (train on the source domain, protect against the target domain):

```
cd Target_Specified
python train_vit_b16.py --dataset officehome --task A-C
python train_vit_b16.py --dataset domainnet  --task C-P
python train_vit_b16.py --dataset office31   --task A-D
```

- `--dataset`: `officehome` (A C P R), `domainnet` (C P R S), `office31` (A D W).
- `--task`: `<source>-<target>` pair; one run per pair, e.g. all 12 Office-Home pairs: `for t in A-C A-P A-R C-A C-P C-R P-A P-C P-R R-A R-C R-P; do python train_vit_b16.py --dataset officehome --task $t; done`
- Default training batch size per dataset (largest that fits a 96 GB GPU): Office-Home 24, Mini-DomainNet 12, Office-31 48. Override with `--batch`; halve it on 48 GB GPUs.
- Other options: `--epoch` (default 10), `--seed` (default 1). Config overrides can be appended as `KEY VALUE`, e.g. `DATALOADER.NUM_WORKERS 4`.
- Results are written per epoch to `output/<dataset>/<task>/seed_<seed>/metric.txt` (source and target accuracy, and the fraction predicted as `unauthorized`); per-epoch checkpoints go to the same directory.

Applicability Authorization by IP-CLIP
```
python Authorization/train_vit_b16.py
```


## Contact
If you have any problem about our code, feel free to contact
- lywang12@126.com
- wangmeng9218@126.com
