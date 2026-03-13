
<p align="right">
  <a href="readme.md">
    <img src="https://img.shields.io/badge/English-View-blue?style=for-the-badge">
  </a>
  <a href="readme_zh.md">
    <img src="https://img.shields.io/badge/中文-阅读-red?style=for-the-badge">
  </a>
</p>

# SAR-Optical-Matching

A deep-learning-based SAR–Optical image registration system, including model training, evaluation, and a visualization GUI.

## Environment Setup

```bash
conda create -n image_reg python=3.10
conda activate image_reg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-image scipy tqdm pyqt5
pip install numpy==1.26.0
```

## Dataset

This project uses the **OS (Optical-SAR) dataset**, which contains paired SAR and optical images.  
Two resolution versions are provided, each split into **train / val / test** subsets:

- **OSdataset/512/**: SAR–Optical image pairs with resolution 512×512  
- **OSdataset/256/**: SAR–Optical image pairs with resolution 256×256 (downsampled from 512)

## Training

1. Use `gen_sar_opt.py` to split 512×512 images in `OSdataset/512` into **64×64 patches** for descriptor training.  
   Modify the dataset paths:

```python
data_root = 'OSdataset/512/'
patch_root = 'OSdataset/patch/'
```

2. Running this script will also generate index files:
   - `OS_train.txt`
   - `OS_val.txt`
   - `OS_test.txt`

3. Modify the paths in `train.py`:

```python
cfg.train_data = 'OS_train.txt'
cfg.test_data = 'OS_val.txt'
cfg.weights_dir = 'weights/'
```

4. Start training:

```bash
python train.py
```

Trained model weights will be saved in the `weights/` directory.

## Evaluation

Evaluation uses the dataset located in the `OS_crop/` directory.  
This directory is cropped from the original dataset.

Each image pair is stored in a folder such as `sar1/`, containing:

- `sar{n}.png` — 512×512 SAR image
- `opt{n}.png` — 480×480 optical image (with ~32 pixel translation offset)
- `mat.txt` — Ground truth transformation matrix between SAR and optical images

Run evaluation using `eval.py` and modify the paths:

```python
eval_path = 'OS_crop'
model_base_path = f'{_model_base_path}/weights/'
```

After evaluation, results such as the following will be printed:

```
mse: 1.8844 1.7377 2.6995 rate 0.9232
```

## GUI Visualization

This project provides a **PyQt5-based GUI** for interactive SAR–Optical image registration.

### Launch

```bash
python Ui_MainWindow.py
```

### Usage

1. **Import SAR image** – click *Import SAR* and select a SAR image file (png/jpg).
2. **Import Optical image** – click *Import OPT* and select an optical image file.
3. **Run registration** – click *Register*. The system will automatically perform feature extraction, feature matching, and homography estimation.
4. **View results**
   - Top-left: SAR image
   - Top-right: Optical image
   - Bottom: registration result with matching lines
   - Right panel: MSE (Mean Squared Error) indicating registration accuracy

### Interface Description

- **SAR image panel (top-left)** — displays the SAR image
- **OPT image panel (top-right)** — displays the optical image
- **Registration result panel (bottom)** — shows stitched images and match lines
- **MSE metric** — lower values indicate better registration accuracy

<img src="UI.png" alt="GUI Interface" width="80%">
