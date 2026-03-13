# SAR-Optical-Matching

[English](./readme.md)

一个基于深度学习的 SAR-光学图像配准系统，包含模型训练、评估以及基于 PyQt5 的可视化 GUI 界面。

## 环境配置

```bash
conda create -n image_reg python=3.10
conda activate image_reg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-image scipy tqdm pyqt5
pip install numpy==1.26.0
```

## 数据集

本项目使用 OS（Optical-SAR）数据集，包含成对的 SAR 与光学图像。数据集提供两种分辨率版本，并且每种分辨率下都划分为 `train`、`val` 和 `test` 三个子集：

- **OSdataset/512/**：512×512 分辨率的 SAR-光学图像对
- **OSdataset/256/**：256×256 分辨率的 SAR-光学图像对（由 512 版本下采样得到）

## 训练

1. 使用 `gen_sar_opt.py` 将 `OSdataset/512/` 中的 512×512 图像切分为 64×64 的 patch，并生成到 `OSdataset/patch/` 目录中，用于训练特征描述子网络。需要先修改脚本中的数据集路径：
   ```python
   data_root = 'OSdataset/512/'
   patch_root = 'OSdataset/patch/'
   ```
2. 运行后会同时生成 `OS_train.txt`、`OS_val.txt` 和 `OS_test.txt` 索引文件。
3. 修改 `train.py` 中的相关路径：
   ```python
   cfg.train_data = 'OS_train.txt'
   cfg.test_data = 'OS_val.txt'
   cfg.weights_dir = 'weights/'
   ```
4. 运行 `python train.py` 开始训练，训练得到的权重文件会保存在 `weights/` 目录下。

## 评估

评估使用 `OS_crop/` 目录下的数据，该目录由原始数据集裁剪得到。每组图像对都存放在一个独立文件夹中（例如 `sar1/`），其中包含：

- `sar{n}.png`：512×512 的 SAR 图像
- `opt{n}.png`：480×480 的光学图像（相对于 SAR 图像存在 32 像素的平移偏移）
- `mat.txt`：真实变换矩阵（ground truth），记录 SAR 与光学图像之间的几何变换关系

使用 `eval.py` 在测试集上进行评估，需先修改模型路径和数据集路径：

```python
eval_path = 'OS_crop'
model_base_path = f'{_model_base_path}/weights/'
```

运行评估脚本后，输出结果示例如下：`mse: 1.8844 1.7377 2.6995 rate 0.9232`

## 可视化界面

本项目提供了一个基于 PyQt5 的可视化 GUI，可用于交互式执行 SAR-光学图像配准。

### 启动方式

```bash
python Ui_MainWindow.py
```

### 使用步骤

1. **导入 SAR 图像**：点击右侧 **Import SAR** 按钮，选择 SAR 图像文件（支持 `png`、`jpg` 等格式）。
2. **导入 OPT 图像**：点击右侧 **Import OPT** 按钮，选择光学图像文件（支持 `png`、`jpg` 等格式）。
3. **执行配准**：点击右侧 **Register** 按钮，系统会自动完成特征提取、特征匹配和单应性矩阵估计。
4. **查看结果**：
   - 左上区域显示 SAR 图像，右上区域显示光学图像。
   - 下方区域显示配准结果，并使用绿色连线标出匹配点对。
   - 右侧显示 MSE（均方误差）数值，用于衡量配准精度。

### 界面说明

- **SAR 图像区域**（左上）：显示导入的 SAR 图像
- **OPT 图像区域**（右上）：显示导入的光学图像
- **配准结果区域**（下方）：显示拼接后的 SAR-光学图像结果及匹配连线
- **MSE 指标**：显示均方误差，数值越小表示配准精度越高

<img src="UI.png" alt="GUI界面" width="80%">
