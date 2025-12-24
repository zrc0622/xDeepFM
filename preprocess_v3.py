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
    Preprocess v3: 低频过滤版 (Low Frequency Filtering)
    1. 稀疏特征 (Sparse): 统计频次，将出现次数 < Threshold 的类别统一替换为 '<UNK>'，然后 LabelEncode。
    2. 稠密特征 (Dense): 保持 V2 的逻辑 (Log1p + MinMaxScaler)。
    """
    cfg = load_config(config_path)
    data_cfg = cfg['data']

    print("Loading raw data")
    file_path = data_cfg['path']
    sample_rows = data_cfg.get('sample_rows', None) 
    
    # 定义列名
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    col_names = ['label'] + dense_features + sparse_features

    # 读取数据
    data = pd.read_csv(file_path, sep='\t', header=None, names=col_names, nrows=sample_rows)

    print("Filling missing values")
    # 缺失值填充
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    print("Encoding features")
    feature_metadata = {
        'sparse': {},
        'dense': dense_features
    }

    # 1. 处理稀疏特征 (Sparse)
    # 设定低频阈值，通常根据数据量级设定
    FREQ_THRESHOLD = 10
    
    for feat in sparse_features:
        data[feat] = data[feat].astype(str)

        # 计算每个类别的出现频次
        value_counts = data[feat].value_counts()

        # 找出频次低于阈值的"长尾"类别索引
        rare_cats = value_counts[value_counts < FREQ_THRESHOLD].index

        # 如果存在低频类别，将它们统一替换为特殊标记 '<UNK>' (Unknown)
        # 作用：
        # 1. 减小 Embedding 矩阵大小，节省显存/内存。
        # 2. 强迫模型学习一个通用的 Embedding 来代表没见过的或罕见的类别，提高对冷启动或长尾数据的泛化能力。
        if len(rare_cats) > 0:
            data.loc[data[feat].isin(rare_cats), feat] = '<UNK>'

        # 编码：将字符串映射为ID
        # 注意：'<UNK>' 也会被分配一个固定的整数 ID
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        
        # 记录词表大小
        feature_metadata['sparse'][feat] = int(data[feat].nunique())

    # 2. 处理稠密特征 (Dense)
    # 保持 V2 的逻辑：截断负值 -> Log变换 -> 归一化
    data[dense_features] = data[dense_features].clip(lower=0)

    data[dense_features] = np.log1p(data[dense_features])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    print(f"Splitting data (test_size={data_cfg['test_size']})")
    # 划分数据集
    train, test = train_test_split(data, test_size=data_cfg['test_size'], random_state=data_cfg['seed'])

    # 设置输出目录为 processed_v3
    processed_dir = "data/processed_v3"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print("Saving processed data to pickle")
    train.to_pickle(os.path.join(processed_dir, 'train.pkl'))
    test.to_pickle(os.path.join(processed_dir, 'test.pkl'))

    with open(os.path.join(processed_dir, 'feature_metadata.pkl'), 'wb') as f:
        pickle.dump(feature_metadata, f)
        
    print(f"Preprocessing Done! Data saved to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xdeepfm.yaml")
    args = parser.parse_args()
    preprocess(args.config)