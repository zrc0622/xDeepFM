import os
import pandas as pd
import numpy as np
import pickle
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_config(config_path):
    """加载 YAML 格式的配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess(config_path):
    """
    Preprocess v2: Log变换增强版
    1. 稀疏特征 (Sparse): 使用 LabelEncoder 编码。
    2. 稠密特征 (Dense): 先进行 Log1p 变换平滑分布，再进行 MinMaxScaler 归一化。
    """
    cfg = load_config(config_path)
    data_cfg = cfg['data']

    print("Loading raw data")
    file_path = data_cfg['path']
    # sample_rows 用于快速调试，正式运行时通常设为 None 读取全量数据
    sample_rows = data_cfg.get('sample_rows', None) 
    
    # 定义 Criteo 数据集的列名结构
    # I1-I13: 13个整数型连续特征 (Dense)
    # C1-C26: 26个类别型特征 (Sparse)
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    col_names = ['label'] + dense_features + sparse_features

    # 读取数据 (Criteo数据集通常是制表符分隔)
    data = pd.read_csv(file_path, sep='\t', header=None, names=col_names, nrows=sample_rows)

    print("Filling missing values")
    # 缺失值处理策略：
    # 类别特征缺失填补为 '-1'，将其视为一个专门的"未知"类别
    data[sparse_features] = data[sparse_features].fillna('-1')
    # 数值特征缺失填补为 0
    data[dense_features] = data[dense_features].fillna(0)

    print("Encoding features")
    # 初始化特征元数据，用于后续模型构建 Embedding 层时指定维度
    feature_metadata = {
        'sparse': {},
        'dense': dense_features
    }

    # 1. 处理稀疏特征 (Sparse)
    for feat in sparse_features:
        lbe = LabelEncoder()
        # 将数据转为字符串，然后映射为整数索引 (0, 1, 2...)
        data[feat] = lbe.fit_transform(data[feat].astype(str))
        # 记录该特征的词表大小 (Vocabulary Size)，例如 C1 有 100 个取值，则记录 100
        feature_metadata['sparse'][feat] = int(data[feat].nunique())

    # 2. 处理稠密特征 (Dense)
    # 截断负值，防止 log 计算出错 (虽然填0后理论上无负值，但作为一种保护机制)
    data[dense_features] = data[dense_features].clip(lower=0)

    # 使用 log(x + 1) 平滑数据分布
    # 原始数值特征往往呈现长尾分布（大部分值很小，极少数值极大），这会影响神经网络的收敛
    # Log 变换可以压缩大值区间，扩展小值区间，使分布接近正态分布
    data[dense_features] = np.log1p(data[dense_features])

    # 归一化
    # 将 Log 变换后的数据缩放到 [0, 1] 区间，消除不同特征量纲的影响
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    print(f"Splitting data (test_size={data_cfg['test_size']})")
    # 划分训练集和测试集
    # random_state 保证每次运行划分一致，便于复现结果
    train, test = train_test_split(data, test_size=data_cfg['test_size'], random_state=data_cfg['seed'])

    # 设置输出目录为 processed_v2
    processed_dir = "data/processed_v2"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print("Saving processed data to pickle")
    # 保存处理好的 DataFrame，Pickle 格式读取效率远高于 CSV
    train.to_pickle(os.path.join(processed_dir, 'train.pkl'))
    test.to_pickle(os.path.join(processed_dir, 'test.pkl'))

    # 保存特征元数据 (字典格式)
    # 训练脚本将读取此文件来决定 Embedding 矩阵的大小
    with open(os.path.join(processed_dir, 'feature_metadata.pkl'), 'wb') as f:
        pickle.dump(feature_metadata, f)
        
    print(f"Preprocessing Done! Data saved to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xdeepfm.yaml")
    args = parser.parse_args()
    preprocess(args.config)