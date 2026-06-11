# DRDD: Decoupled Residual Denoising Diffusion Models for Unified and Data-Efficient Image-to-Image Translation

Official PyTorch implementation of **Decoupled Residual Denoising Diffusion Models (DRDD)** for unified and data-efficient image-to-image (I2I) translation.

DRDD decouples the conventional diffusion-based I2I process into two stages:

1. **Noise diffusion** for domain harmonization and manifold lifting.
2. **Residual diffusion** for deterministic source-to-target semantic mapping in the fixed-noise domain.

![Main figure](../pics/59f95e98bdb81b5b8fa4da76e4c9b40f.png)

## News

- Pre-trained model and inference code are available.
- Training code is included, but users should verify dataset paths and training configuration before running.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Dataset Preparation](#dataset-preparation)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Training](#training)
- [Repository Structure](#repository-structure)
- [Common Issues](#common-issues)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Requirements

The code has been tested with the following environment:

- Linux
- NVIDIA GPU with CUDA support
- Conda
- Python 3.7+
- PyTorch 1.13.1 with CUDA 11.6

## Installation

Clone the repository and enter the code directory:

```bash
git clone https://github.com/HKU-HealthAI/DRDD.git
cd DRDD/DRDD-code
```

Create and activate the Conda environment:

```bash
conda env create -f install.yaml
conda activate drdd
```

If `mpi4py` or distributed utilities fail to import, install OpenMPI first:

```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin
```

## Pre-trained Models

Download the pre-trained checkpoint and place it under `./ckpt/`:

```text
DRDD-code/
└── ckpt/
    └── DRDD_AiO5.pt
```

Available download link:

- Quark: https://pan.quark.cn/s/8d7316f9684e
- Extraction code: `iXth`

After downloading, make sure the checkpoint path in `config/AiO5_test.yaml` matches your local file:

```yaml
test:
  results_ckpt: "./ckpt/DRDD_AiO5.pt"
```

## Dataset Preparation

Place all datasets under `DRDD-code/dataset/`.

Expected directory structure:

```text
DRDD-code/
└── dataset/
    ├── denoise/
    │   ├── CBSD68/
    │   └── make_noise_dataset.py
    ├── Rain100L/
    │   ├── input/
    │   └── target/
    ├── dehaze/
    │   ├── gt/
    │   ├── haze/
    │   └── make_dehaze_flist.py
    ├── GoPro/
    │   ├── input/
    │   └── target/
    └── lol/
        ├── high/
        └── low/
```

Supported All-in-One-5 tasks:

| Task | Input folder | Target / GT folder |
| --- | --- | --- |
| Deraining | `dataset/Rain100L/input` | `dataset/Rain100L/target` |
| Deblurring | `dataset/GoPro/input` | `dataset/GoPro/target` |
| Dehazing | `dataset/dehaze/haze` | `dataset/dehaze/gt` |
| Denoising | generated noisy CBSD68 inputs | `dataset/denoise/CBSD68` |
| Low-light enhancement | `dataset/lol/low` | `dataset/lol/high` |

### Generate auxiliary test files

Generate the denoising test dataset:

```bash
python ./dataset/denoise/make_noise_dataset.py
```

Generate the dehazing file lists:

```bash
python ./dataset/dehaze/make_dehaze_flist.py
```

## Inference

Edit the task field in `config/AiO5_test.yaml`:

```yaml
test:
  task: light   # choose from: light, rain, fog, noise, blur
  test_result_folder: "./Experience_record/aio5/light"
```

Then run inference:

```bash
python test.py --data_config AiO5_test
```

The default output directory is:

```text
./Experience_record/aio5/<task>/test/dataset/
```

For example, when `task: rain`, predictions are expected under:

```text
./Experience_record/aio5/rain/test/dataset/
```

## Evaluation

Use `evaluation.py` to compute PSNR, SSIM, LPIPS, and optionally NIQE.

### Deraining

```bash
python evaluation.py \
  --gt_dir ./dataset/Rain100L/target \
  --pred_dir ./Experience_record/aio5/rain/test/dataset/ \
  --output rain_metrics.csv \
  --calculate_lpips
```

### Low-light enhancement

```bash
python evaluation.py \
  --gt_dir ./dataset/lol/high \
  --pred_dir ./Experience_record/aio5/light/test/dataset/ \
  --output light_metrics.csv \
  --calculate_lpips
```

### Dehazing

```bash
python evaluation.py \
  --gt_dir ./dataset/dehaze/gt \
  --pred_dir ./Experience_record/aio5/haze/test/dataset/ \
  --output haze_metrics.csv \
  --calculate_lpips
```

### Denoising

```bash
python evaluation.py \
  --gt_dir ./dataset/denoise/CBSD68 \
  --pred_dir ./Experience_record/aio5/noise/test/dataset/ \
  --output noise_metrics.csv \
  --calculate_lpips
```

### Deblurring

```bash
python evaluation.py \
  --gt_dir ./dataset/GoPro/target \
  --pred_dir ./Experience_record/aio5/blur/test/dataset/ \
  --output blur_metrics.csv \
  --calculate_lpips
```

## Training

Training entry point:

```bash
python train_noIn.py --data_config train_diffuir
```

Before training, check `config/train_diffuir.yaml` carefully and replace any local absolute paths with your own dataset path, for example:

```yaml
data:
  data_root: "./dataset"
```

Training currently constructs multiple task datasets in `train_noIn.py`. Make sure all required task folders and file lists exist before launching training.

## Repository Structure

```text
DRDD/
├── README.md
├── pics/
└── DRDD-code/
    ├── config/
    │   ├── AiO5_test.yaml
    │   └── train_diffuir.yaml
    ├── data/
    ├── metrics/
    ├── src/
    ├── evaluation.py
    ├── install.yaml
    ├── test.py
    └── train_noIn.py
```

## Common Issues

### 1. `FileNotFoundError: ./config/<name>.yaml`

Make sure you run commands from `DRDD-code/`, not the repository root.

Correct:

```bash
cd DRDD/DRDD-code
python test.py --data_config AiO5_test
```

### 2. Checkpoint not found

Check the path below in `config/AiO5_test.yaml`:

```yaml
test:
  results_ckpt: "./ckpt/DRDD_AiO5.pt"
```

### 3. Dataset path not found

Check the dataset root in the config file:

```yaml
data:
  data_root: "./dataset"
```

### 4. Wrong task output folder

When switching `test.task`, also update `test.test_result_folder` accordingly.

Example:

```yaml
test:
  task: rain
  test_result_folder: "./Experience_record/aio5/rain"
```

## Acknowledgements

This repository is based on the following excellent projects:

- [RDDM](https://github.com/nachifur/RDDM)
- [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR)
- [guided-diffusion](https://github.com/openai/guided-diffusion)

We thank the authors for their contributions to the community.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{lin2026drdd,
  title={Decoupled Residual Denoising Diffusion Models for Unified and Data Efficient Image-to-Image Translation},
  author={Lin, Ziyue and Hou, Jiahe and Xia, Hongyu and Xie, Xinrui and Wang, Feifei and Zhou, Yuyin and Wang, Wei and Liu, Jiawei and Qu, Liangqiong},
  journal={arXiv preprint arXiv:2606.01048},
  year={2026}
}
```
