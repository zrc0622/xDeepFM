import os
import pandas as pd
import pickle
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_config(config_path):
    # 加载YAML配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess(config_path):
    """
    Preprocess v1: 基础版本
    1. 稀疏特征 (Sparse): 直接使用 LabelEncoder 编码，不做低频过滤。
    2. 稠密特征 (Dense): 直接使用 MinMaxScaler 归一化，不做对数变换 (Log Transform)。
    """
    cfg = load_config(config_path)
    data_cfg = cfg['data']

    print("Loading raw data")
    file_path = data_cfg['path']
    # sample_rows 用于调试，若配置中有定义则只读取部分行，否则读取全部
    sample_rows = data_cfg.get('sample_rows', None) 
    
    # 定义 Criteo 数据集的列名
    # I1-I13: 稠密特征 (Integer/Continuous)
    # C1-C26: 稀疏特征 (Categorical)
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    col_names = ['label'] + dense_features + sparse_features

    # 读取原始数据 (Tab分割)
    data = pd.read_csv(file_path, sep='\t', header=None, names=col_names, nrows=sample_rows)

    print("Filling missing values")
    # 缺失值填充策略：
    # 稀疏特征填充字符串 '-1' (作为一个独立的类别)
    data[sparse_features] = data[sparse_features].fillna('-1')
    # 稠密特征填充 0
    data[dense_features] = data[dense_features].fillna(0)

    print("Encoding features")
    # 初始化元数据字典，用于后续训练时构建 Embedding 层
    feature_metadata = {
        'sparse': {},
        'dense': dense_features
    }

    # 1. 处理稀疏特征 (Sparse Features)
    # v1策略：直接对所有出现的类别进行编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        # 将数据转为字符串后编码 (0, 1, 2, ...)
        data[feat] = lbe.fit_transform(data[feat].astype(str))
        # 记录特征的词表大小 (Vocabulary Size)
        feature_metadata['sparse'][feat] = int(data[feat].nunique())

    # 2. 处理稠密特征 (Dense Features)
    # v1策略：直接进行 MinMax 归一化，不进行 Log 变换
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    print(f"Splitting data (test_size={data_cfg['test_size']})")
    # 划分训练集和测试集
    train, test = train_test_split(data, test_size=data_cfg['test_size'], random_state=data_cfg['seed'])

    # 设置输出目录为 processed_v1
    processed_dir = "data/processed_v1"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print("Saving processed data to pickle")
    # 使用 pickle 格式保存，读取速度比 CSV 快得多，且保留了数据类型
    train.to_pickle(os.path.join(processed_dir, 'train.pkl'))
    test.to_pickle(os.path.join(processed_dir, 'test.pkl'))

    # 保存特征元数据
    with open(os.path.join(processed_dir, 'feature_metadata.pkl'), 'wb') as f:
        pickle.dump(feature_metadata, f)
        
    print(f"Preprocessing Done! Data saved to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xdeepfm.yaml")
    args = parser.parse_args()
    preprocess(args.config)