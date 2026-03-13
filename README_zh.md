# SAR-Optical-Registration

<p align="left">
  <a href="README.md">
    <img src="https://img.shields.io/badge/English-red" height="22">
  </a>
</p>

基于深度学习的SAR-光学图像配准系统，包含模型训练、评估以及可视化GUI界面。

## 环境安装

```bash
conda create -n image_reg python=3.10
conda activate image_reg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-image scipy tqdm pyqt5
pip install numpy==1.26.0
```

## 数据集

本项目使用OS（Optical-SAR）数据集，包含SAR和光学图像对，提供两种分辨率版本，每种分辨率下分为 train、val、test 三个子集：

- **OSdataset/512/**：512x512 分辨率的SAR-光学图像对
- **OSdataset/256/**：256x256 分辨率的SAR-光学图像对（512版本的降采样）

## 训练

1. 使用 `gen_sar_opt.py` 将 OSdataset/512 的512x512图像切分为64x64的patch，生成到 `OSdataset/patch/` 目录，用于训练特征描述子网络。修改其中的数据集路径：
   ```python
   data_root = 'OSdataset/512/'
   patch_root = 'OSdataset/patch/'
   ```
2. 运行后会同时生成 `OS_train.txt`、`OS_val.txt`、`OS_test.txt` 索引文件
3. 修改 `train.py` 里面的相关路径：
   ```python
   cfg.train_data = 'OS_train.txt'
   cfg.test_data = 'OS_val.txt'
   cfg.weights_dir = 'weights/'
   ```
4. `python train.py` 开始进行模型训练，训练好的权重文件会保存在 `weights/` 目录下

## 评估

评估使用 `OS_crop/` 目录下的数据，该目录由原始数据集裁剪而来。每组图像对存放在独立文件夹中（如 `sar1/`），包含：

- `sar{n}.png`：512x512 的SAR图像
- `opt{n}.png`：480x480 的光学图像（相对SAR图像存在32像素的平移偏移）
- `mat.txt`：真实变换矩阵（ground truth），记录了SAR与光学图像之间的几何变换关系

使用 `eval.py` 脚本进行测试集评估，修改模型路径和数据集路径：

```python
eval_path = 'OS_crop'
model_base_path = f'{_model_base_path}/weights/'
```

运行评估，在 OS 数据集（90 组图像对）上的最佳结果如下：

- **MSE（均方误差）**：匹配特征点在 x、y 及 xy 方向上的均方根误差（单位：像素），越低越好。
- **RCM（正确匹配率）**：误差低于阈值的匹配特征点占比，越高越好。

<table align="center">
  <tr>
    <th colspan="3" align="center">MSE</th>
    <th rowspan="2" align="center">RCM</th>
  </tr>
  <tr>
    <td align="center">xMSE</td>
    <td align="center">yMSE</td>
    <td align="center">xyMSE</td>
  </tr>
  <tr>
    <td align="center">1.672</td>
    <td align="center">1.841</td>
    <td align="center">2.6003</td>
    <td align="center">92.8%</td>
  </tr>
</table>

## 可视化界面

本项目提供了基于PyQt5的可视化GUI界面，可以交互式地进行SAR-光学图像配准操作。

### 启动方式

```bash
python Ui_MainWindow.py
```

### 使用步骤

1. **导入SAR图像**：点击右侧"导入SAR"按钮，选择SAR图像文件（支持png、jpg格式）
2. **导入OPT图像**：点击右侧"导入OPT"按钮，选择光学图像文件（支持png、jpg格式）
3. **执行配准**：点击右侧"配准"按钮，系统将自动进行特征提取、特征匹配和单应性矩阵估计
4. **查看结果**：
   - 上方左侧显示SAR图像，右侧显示光学图像
   - 下方显示配准结果，包含匹配点对之间的连线（绿色线条标注匹配关系）
   - 右侧显示MSE（均方误差）数值，用于评估配准精度

### 界面说明

- **SAR图像区域**（左上）：显示导入的SAR图像
- **OPT图像区域**（右上）：显示导入的光学图像
- **配准结果区域**（下方）：显示SAR和光学图像的拼接结果及匹配连线
- **MSE指标**：显示配准的均方误差，数值越小表示配准精度越高

### 效果展示

<img src="demo.png" alt="demo" width="80%">
