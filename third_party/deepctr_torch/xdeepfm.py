# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, CIN


class xDeepFM(BaseModel):
    """
    xDeepFM 架构实例化。
    
    参数说明:
    :param linear_feature_columns: 用于线性部分(Linear)的特征列。
    :param dnn_feature_columns: 用于深度部分(Deep/CIN)的特征列。
    :param dnn_hidden_units: DNN 部分每层的神经元数量，如 (256, 128)。
    :param cin_layer_size: CIN 部分每层特征图的数量，如 (128, 128)。
    :param cin_split_half: CIN 的一种变体配置，是否将特征图减半。
    :param cin_activation: CIN 特征图的激活函数。
    :param l2_reg_*: 各部分的 L2 正则化系数。
    :param init_std: 初始化标准差。
    :param dnn_dropout: DNN 的 Dropout 比率。
    :param dnn_use_bn: DNN 是否使用 BatchNormalization。
    :param task: "binary" (二分类) 或 "regression" (回归)。
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        # 1. 调用父类 BaseModel 初始化
        # BaseModel 负责创建 embedding_dict (用于DNN/CIN) 和 linear_model (用于Linear部分)
        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        
        # 2. 初始化 DNN 部分 (隐式高阶交互)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            # 计算 DNN 输入维度 (所有 Embedding 展平后的长度 + 稠密特征数)
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            # DNN 输出层：映射到 1 维 Logit
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            
            # 添加正则化
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        # 3. 初始化 CIN 部分 (显式高阶交互 - Compressed Interaction Network)
        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict) # 特征域的数量
            # 计算 CIN 输出的总特征图数量 (Sum Pooling 前的维度)
            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            
            # 实例化 CIN
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            # CIN 输出层
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)

        self.to(device)

    def forward(self, X):
        """
        前向传播逻辑
        X: 输入 Tensor [batch_size, total_features]
        """
        
        # 1. 提取 Embedding 和 稠密特征
        # sparse_embedding_list: [feat1_emb, feat2_emb, ...] -> List of [batch, 1, emb_dim]
        # dense_value_list: [feat1_val, feat2_val, ...] -> List of [batch, 1]
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        # 2. Linear 部分 (Wide)
        # 类似于 Logistic Regression，处理记忆能力
        linear_logit = self.linear_model(X)
        
        # 3. CIN 部分 (Cross)
        # 显式向量级交互
        if self.use_cin:
            # 拼接 Embedding: [batch, field_num, embed_dim]
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
            
        # 4. DNN 部分 (Deep)
        # 隐式比特级交互
        if self.use_dnn:
            # 展平并拼接所有输入
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        # 5. 融合各部分输出
        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:  # linear + CIN
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:  # linear + Deep
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + CIN + Deep (Standard xDeepFM)
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError

        # 6. 输出预测值 (Sigmoid for binary, Identity for regression)
        y_pred = self.out(final_logit)

        return y_pred