# Image_Registration

基于深度学习的SAR-光学图像配准系统，包含模型训练、评估以及可视化GUI界面。

## 环境安装

本代码依赖pytorch环境，在有nvidia显卡的条件下

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

否则需安装

```bash
pip3 install torch torchvision torchaudio
```

其他的python依赖有

- opencv-python
- scikit-image
- scipy
- numpy == 1.26.0
- tqdm
- pyqt5

## 数据集

本项目使用OS（Optical-SAR）数据集，数据集不包含在本仓库中，需要单独下载。

数据集下载地址：<!-- TODO: 请补充数据集下载链接 -->

下载后请将数据集放置在项目根目录下，目录结构如下：

```
OSdataset/
├── 256/
│   ├── train/    # 256x256 训练集（包含opt和sar图像对）
│   ├── val/      # 256x256 验证集
│   └── test/     # 256x256 测试集
└── 512/
    ├── train/    # 512x512 训练集（包含opt和sar图像对）
    ├── val/      # 512x512 验证集
    └── test/     # 512x512 测试集
```

## 训练

1. 先使用 `gen_sar_opt.py` 生成匹配数据集，生成的图片大小为64x64
2. 修改 `gen_sar_opt.py` 中的数据集路径：
   ```python
   data_root = 'OSdataset/512/'
   patch_root = 'OSdataset/patch/'
   ```
3. 运行后会生成 `OS_train.txt` 和 `OS_val.txt` 用于训练
4. 修改 `train.py` 里面的相关路径：
   ```python
   cfg.train_data = 'OS_train.txt'
   cfg.test_data = 'OS_val.txt'
   cfg.weights_dir = 'weights/'
   ```
5. `python train.py` 开始进行模型训练，训练好的权重文件会保存在 `weights/` 目录下

## 评估

1. 使用 `eval.py` 脚本进行测试集评估
2. 修改模型路径和数据集路径：
   ```python
   eval_path = 'OS_crop'
   model_base_path = f'{_model_base_path}/weights/'
   ```
3. 开始运行评估，评估完成会有结果输出：`mse: 1.8844 1.7377 2.6995 rate 0.9232`

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
