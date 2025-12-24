import os
import pandas as pd
import numpy as np
import pickle
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess(config_path):
    """
    Preprocess v4: 数值特征离散化 (Discretization/Binning)
    1. 稀疏特征 (Sparse): 采用 V3 的策略，低频过滤 (<UNK>) + LabelEncoding。
    2. 稠密特征 (Dense): Log变换 -> 分箱 (Binning) -> 转化为稀疏特征 ID。
       结果是所有特征都被视为稀疏特征处理，学习 Embedding。
    """
    cfg = load_config(config_path)
    data_cfg = cfg['data']

    print("Loading raw data")
    file_path = data_cfg['path']
    sample_rows = data_cfg.get('sample_rows', None) 
    
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    col_names = ['label'] + dense_features + sparse_features

    data = pd.read_csv(file_path, sep='\t', header=None, names=col_names, nrows=sample_rows)

    print("Filling missing values")
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    print("Encoding features")
    feature_metadata = {
        'sparse': {},
        'dense': [] 
    }

    # 1. 处理稀疏特征
    print("Processing Sparse Features")
    FREQ_THRESHOLD = 10
    for feat in sparse_features:
        data[feat] = data[feat].astype(str)

        value_counts = data[feat].value_counts()
        rare_cats = value_counts[value_counts < FREQ_THRESHOLD].index

        if len(rare_cats) > 0:
            data.loc[data[feat].isin(rare_cats), feat] = '<UNK>'

        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        feature_metadata['sparse'][feat] = int(data[feat].nunique())

    # 2. 处理稠密特征 (Binning)
    print("Processing Dense Features (Log + Binning)")
    
    # a. 预处理：截断与 Log 变换 (平滑分布，使分箱更均匀)
    data[dense_features] = data[dense_features].clip(lower=0)
    data[dense_features] = np.log1p(data[dense_features])

    # b. 分箱操作
    n_bins = 100 # 将数值范围划分为 100 个区间
    for feat in dense_features:
        # pd.cut: 将连续数值切分为 n_bins 个桶
        # labels=False: 返回桶的整数索引 (0 ~ 99)
        # include_lowest=True: 确保最小值也能被包含在第一个桶中
        data[feat] = pd.cut(data[feat], bins=n_bins, labels=False, include_lowest=True)

        # 确保类型为整数
        data[feat] = data[feat].astype(int)

        # c. 更新元数据
        # 将原数值特征记录为稀疏特征，词表大小为桶的数量 (最大索引 + 1)
        # 这样模型会为每个"桶"学习一个 Embedding 向量
        feature_metadata['sparse'][feat] = int(data[feat].max() + 1)

    print(f"Splitting data (test_size={data_cfg['test_size']})")
    train, test = train_test_split(data, test_size=data_cfg['test_size'], random_state=data_cfg['seed'])

    # 设置输出目录为 processed_v4
    processed_dir = "data/processed_v4"
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