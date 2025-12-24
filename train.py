import os
import sys
import yaml
import pickle
import shutil
import datetime
import argparse
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import xDeepFM
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

# 自定义 Logger 类，用于将 print 输出同时重定向到控制台和日志文件
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_feature_columns(feature_metadata, embedding_size):
    """
    根据预处理生成的元数据构建 DeepCTR-Torch 的 Feature Columns
    """
    sparse_cols = feature_metadata['sparse'] # 字典: {'FeatureName': vocab_size, ...}
    dense_cols = feature_metadata['dense']   # 列表: ['FeatureName', ...]

    # 构建特征列列表
    # 1. 稀疏特征：使用 SparseFeat，需指定词表大小和 Embedding 维度
    # 2. 稠密特征：使用 DenseFeat，维度通常为 1
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=embedding_size)
        for feat, vocab_size in sparse_cols.items()
    ] + [
        DenseFeat(feat, 1)
        for feat in dense_cols
    ]
    
    return fixlen_feature_columns

def train(config_path):
    # 1. 加载配置
    cfg = load_config(config_path)
    processed_dir = cfg['data']["processed_dir"]

    # 2. 准备输出目录和日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = cfg['model']['name']
    output_path = os.path.join("outputs", f"{model_name}_{timestamp}")
    if not os.path.exists(output_path): os.makedirs(output_path)
    # 备份配置文件，便于后续查阅实验参数
    shutil.copy(config_path, os.path.join(output_path, "config.yaml"))
    # 重定向标准输出到日志文件
    sys.stdout = Logger(os.path.join(output_path, "train.log"))

    print(f"Config loaded: {config_path}")

    # 3. 加载预处理后的数据 (Pickle格式)
    print("Loading processed data and metadata")
    train_df = pd.read_pickle(os.path.join(processed_dir, 'train.pkl'))
    test_df = pd.read_pickle(os.path.join(processed_dir, 'test.pkl'))
    
    # 加载元数据，这是连接预处理和模型定义的关键
    with open(os.path.join(processed_dir, 'feature_metadata.pkl'), 'rb') as f:
        feature_metadata = pickle.load(f)

    # 4. 构建模型输入定义
    model_cfg = cfg['model']
    feature_columns = get_feature_columns(feature_metadata, model_cfg['embedding_size'])
    
    # xDeepFM 中，通常 Linear 部分和 Deep/CIN 部分共享同样的特征输入
    linear_feature_columns = feature_columns
    dnn_feature_columns = feature_columns
    
    # 获取所有特征名列表
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 5. 准备输入数据字典
    # DeepCTR-Torch 接收的输入是一个字典: {feature_name: column_data}
    train_model_input = {name: train_df[name] for name in feature_names}
    test_model_input = {name: test_df[name] for name in feature_names}
    
    train_label = train_df['label'].values
    test_label = test_df['label'].values

    # 设备检测
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 6. 初始化 xDeepFM 模型
    model = xDeepFM(linear_feature_columns=linear_feature_columns, 
                    dnn_feature_columns=dnn_feature_columns, 
                    task='binary',
                    # 线性部分正则化
                    l2_reg_linear=model_cfg['l2_reg_linear'],
                    # Embedding正则化 (防止过拟合稀疏特征)
                    l2_reg_embedding=model_cfg['l2_reg_embedding'], 
                    # DNN (Deep Part) 结构配置
                    dnn_hidden_units=model_cfg['dnn_hidden_units'], 
                    l2_reg_dnn=model_cfg['l2_reg_dnn'],
                    dnn_dropout=model_cfg['dnn_dropout'],
                    # CIN (Compressed Interaction Network) 结构配置
                    cin_layer_size=model_cfg['cin_layer_size'], 
                    cin_split_half=model_cfg['cin_split_half'], 
                    cin_activation=model_cfg['cin_activation'], 
                    l2_reg_cin=model_cfg['l2_reg_cin'],
                    # DNN 激活函数
                    dnn_activation=model_cfg['dnn_activation'],
                    device=device)

    # 7. 编译模型
    train_cfg = cfg['train']
    if train_cfg['optimizer'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    elif train_cfg['optimizer'] == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'])
        
    model.compile(optimizer=optim, loss="binary_crossentropy", metrics=['binary_crossentropy', 'auc'])

    best_model_path = os.path.join(output_path, "best_model.pth")
    
    # 8. 设置回调函数
    callbacks = [
        # 监控验证集AUC，仅保存最优模型
        ModelCheckpoint(filepath=best_model_path, monitor='val_auc', mode='max', save_best_only=True),
        # 早停机制：如果验证集AUC在3个epoch内没有提升，则停止训练
        EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=False)
    ]

    print("Start Training")
    # 9. 开始训练
    history = model.fit(train_model_input, train_label,
                        batch_size=train_cfg['batch_size'],
                        epochs=train_cfg['epochs'],
                        verbose=2,
                        validation_split=cfg['data']['valid_size'], # 从训练集中划分一部分作为验证集
                        callbacks=callbacks)

    # 保存训练历史数据 (Loss, AUC变化曲线)
    pd.DataFrame(history.history).to_csv(os.path.join(output_path, "training_metrics.csv"), index=False)
    
    # 10. 加载最优模型并评估
    # 必须重新加载 best_model.pth，因为最后一次训练的模型不一定是最优的
    model.load_state_dict(torch.load(best_model_path).state_dict())

    print("Evaluating")
    pred_ans = model.predict(test_model_input, batch_size=train_cfg['batch_size'])
    test_auc = roc_auc_score(test_label, pred_ans)
    test_logloss = log_loss(test_label, pred_ans) 
    print(f"Test AUC: {round(test_auc, 4)}")
    print(f"Test LogLoss: {round(test_logloss, 4)}")

    # 将最优模型重命名，带上AUC分数以便区分
    torch.save(model.state_dict(), os.path.join(output_path, f"{model_name}_auc_{test_auc:.4f}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/xdeepfm.yaml")
    args = parser.parse_args()
    train(args.config)