# 介绍
本项目基于 [deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch) 框架，对经典的 CTR 预估模型 xDeepFM 进行了复现。项目对代码中的关键功能模块（如数据预处理的流程、创建特征列、CIN 模块等，包括 deepctr-torch 的源码）进行了详细的中文注释。

## 文件结构说明

```text
.
├── configs/
│   └── xdeepfm.yaml        # 实验配置文件
├── third_party/            
│   └── deepctr_torch/      # 带注释的 deepctr-torch 源码
│       ├── inputs.py       # 特征列定义、Embedding 构建与索引切片
│       ├── xdeepfm.py      # xDeepFM 模型主体 (Linear + DNN + CIN)
│       ├── cin.py          # CIN 模块的实现
│       └── callback.py     # 训练回调工具
├── eda.py                  # 数据探索分析代码
├── preprocess_v[1-4].py    # 四种不同策略的数据预处理代码
├── train.py                # 模型训练主代码
└── utils.ipynb             # 训练曲线绘制和特征元数据查看
```

<!-- ### deepctr-torch 源码
`third_party/deepctr_torch`目录包含复制自 deepctr-torch 的核心代码，并进行了详细注释：
* `inputs.py`: 负责特征列的定义、Embedding 矩阵的构建以及输入 Tensor 的自动切片与索引映射
* `xdeepfm.py`: xDeepFM 模型的主体结构，定义了 Linear、DNN 与 CIN 三部分的并行逻辑
* `cin.py`: 实现了压缩交互网络（Compressed Interaction Network, CIN）
* `callback.py`: 训练回调工具，包含 `EarlyStopping` 和 `ModelCheckpoint`
### 实验与工具脚本
根目录下为实现的实验与工具脚本：
* `eda.py`: 探索性数据分析，分析标签分布、数值特征相关性、类别特征基数以及缺失值情况
* `preprocess_v1~v4.py`: 不同版本的数据预处理策略
* `train.py`: 训练脚本，支持动态加载配置、自动创建实验日志、模型最优权重保存及测试集评估
* `utils.py`: 训练曲线绘制和特征元数据检查
* `configs/xdeepfm.yaml`: 实验配置文件 -->

## 预处理策略对比

本项目实现了四种递进的预处理策略，以应对 Criteo 数据集的长尾分布和高维稀疏特性：

| Version | Strategy | #Fields | #Features (Sparse) | #Features (Dense) |
| :---: | :---: | :---: | :---: | :---: |
| v1 | Standard | 26 | 33,762,577 | 13 |
| v2 | Log Norm | 26 | 33,762,577 | 13 |
| v3 | Log Norm + Frequency Filtering (<10) | 26 | 1,085,733 | 13 |
| v4 | Discretization (100 bins) + Frequency Filtering (<10)  | 39 | 1,087,033 | 0 |

# 环境配置
```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install deepctr-torch, pyyaml, pandas, numpy==1.26.4, matplotlib, seaborn
```

# 快速开始

下载[`Criteo`](https://cloud.tsinghua.edu.cn/d/72a3745d59bc4c2d84ae/)广告数据集并解压至`data`文件夹下
<!-- bigdatathu -->

## 数据探索
```bash
python eda.py --config configs/xdeepfm.yaml
```

## 数据预处理

```bash
python preprocess_v4.py --config configs/xdeepfm.yaml
```

## 模型训练
```bash
python train.py --config configs/xdeepfm.yaml
```

# 超参数配置

| 参数 | 值 |
| :---: | :---: |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| L2 Regularization | 0.0001 |
| DNN Structure | [400, 400] |
| DNN Activation | ReLU |
| CIN Structure | [200, 200, 200] |
| CIN Activation | Identity |
| Embedding Size | 10 |
| Batch Size | 4096 |
| Max Epochs | 8 |
| Dropout | 0 |
| Dataset Split | 8 : 1 : 1 |

# 实验结果
![result](result.png)
