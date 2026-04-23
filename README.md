# PanoCount: Latitude-Aware Crowd Counting in 360° Panoramic Imagery

[![Paper](https://img.shields.io/badge/Paper-TCSVT-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-Coming%20Soon-orange)]()

Official implementation of **"Beyond Perspective: Latitude-Aware Crowd Counting in Equirectangular Panoramic Imagery"**.

## Overview
This repository contains:
- CSRNet-LP model implementation (latitude prior + circular padding)
- MCNN, CAN, CSRNet baseline implementations
- PanoCount dataset loader and split utilities
- Interactive GUI for inference on 360° images
- Pre-trained model checkpoints

## Dataset
PanoCount will be publicly released upon paper acceptance. To request early access for review, contact the corresponding authors.

## Checkpoints
Pre-trained weights are provided in `checkpoints/`. Due to GitHub file size limits, download them from [release page](../../releases).

## Installation
```bash
conda env create -f environment.yml
conda activate panocount
pip install -r requirements.txt
```

## Training
```bash
python main.py --model csrnet_pano --config configs/csrnet_pano.yaml
```

## Evaluation
```bash
python main.py --model csrnet_pano --eval --checkpoint checkpoints/csrnet_lp_best.pth
```

## GUI
```bash
python gui/app.py
```

---

# PanoCrowdCount

Small starter project for panoramic crowd counting.

What I added for now:
- `test_gpu.py` — a small Python script to check CUDA availability and run a tiny GPU benchmark (matrix multiply + allocation). Useful to confirm your RTX GPU is seen by PyTorch.

Quick start (Windows PowerShell):

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install the dependencies (example):

```powershell
pip install -r requirements.txt
```

3. Run the GPU test:

```powershell
python .\scripts\test_gpu.py
```

## PanoCount Analyzer (GUI)

Desktop app for ERP visualization and multi-model comparison (MCNN, CAN, CSRNet, CSRNet-LP).

```powershell
pip install -r requirements.txt
pip install -r requirements-gui.txt
python .\main.py
```

See [README-analyzer.md](README-analyzer.md) for details. Use a working PyTorch install (e.g. conda env `seg` on Windows if the venv `torch` DLL fails to load).

Notes:
- If `torch` isn't installed with a CUDA-enabled build compatible with your drivers, the script will tell you and exit. Install a matching PyTorch wheel from https://pytorch.org/get-started/locally/.
- The script prints the device name and memory; if it shows your RTX 5070Ti (or similar), your GPU is visible to PyTorch.
