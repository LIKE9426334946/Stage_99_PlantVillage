# Stage_99_PlantVillage

使用 torchvision ViT-B/16 进行 PlantVillage 植物病害图像分类。

## 安装与运行

```bash
pip install -r requirements.txt
python dataset.py
python train.py
```

运行前修改 `config.yaml` 中的 `dataset.root`，指向本机 PlantVillage 目录。默认保留现有参数：
10 个 epoch、batch size 4、学习率 0.0001、AdamW weight decay 0.01、`num_workers: 0`。
`training.device: auto` 自动选择 CUDA 或 CPU，也可以设置为 `cpu` 或 `cuda:0`。

## 配置文件

所有可调整配置集中在 `config.yaml`，各脚本共用 `config.py` 读取，不再在 Python 文件中单独设置参数。

| 分组 | 内容 |
| --- | --- |
| `model` | 模型名称、是否预训练、权重名称、类别数 |
| `training` | 随机种子、轮数、批大小、学习率、权重衰减、加载线程、设备、输出文件和日志间隔 |
| `dataset` | 数据集根目录、图像目录、官方划分文件、通道数、输入尺寸、归一化和图像扩展名 |
| `download` | Hugging Face 数据源及加载选项；下载保存到 `dataset.root` |
| `checks` | 数据检查、抽样数量、输出路径、绘图大小和显示选项 |

数据集保持官方目录和划分方式：

```text
PlantVillage/
  raw/color/<类别名称>/<图像文件>
  splits/color_train.txt
  splits/color_test.txt
  leaf_grouping/leaf-map.json
```

`train_split`、`test_split`、`image_dir` 和 `leaf_map` 相对于 `dataset.root`。
其他相对路径（包括 `dataset.root`、训练输出目录和检查输出目录）相对于所选 YAML 文件所在目录，
因此可以从任意工作目录启动脚本。Windows 路径建议使用单引号，例如 `'D:\04_Datasets\PlantVillage'`。
Linux 上请将 `dataset.root` 改为实际的 Linux 路径。

当前模型名称为 `vit_b_16`，输入为 RGB 三通道正方形，边长须为 16 的整数倍。
默认预训练权重 `IMAGENET1K_V1` 要求 `image_size: [224, 224]`；
关闭 `model.pretrained` 后可以使用其他满足条件的尺寸。
`model.num_classes` 应与数据目录中的类别数相同，训练前会检查是否一致。
启用 `training.deterministic` 时固定随机种子并启用 cuDNN 确定性模式。

所有入口均支持选择另一份 YAML 文件：

```bash
python train.py --config path/to/config.yaml
python dataset.py --config path/to/config.yaml
python model.py --config path/to/config.yaml
python download_dataset.py --config path/to/config.yaml
python test.py --config path/to/config.yaml
python checks/check_dataset.py --config path/to/config.yaml
```

`model.py` 检查输入输出形状；`test.py` 保留 Hugging Face 数据加载检查功能。
`checks/check_dataset.py` 读取本地数据，检查官方划分并保存样例图和类别分布图。
设置 `checks.show_plots: false` 可以只保存图像而不弹出窗口。

训练记录默认保存为 `outputs/metrics.csv`，最后一轮模型保存为 `outputs/last.pth`。
checkpoint 中包含类别名称和本次配置，便于核对实验参数。

配置和数据流回归检查（使用临时样例图像，不下载完整数据集或预训练权重）：

```bash
python -m unittest discover -s tests -v
```
