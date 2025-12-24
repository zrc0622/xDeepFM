# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
"""

from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from .layers.sequence import SequencePoolingLayer
from .layers.utils import concat_fun

DEFAULT_GROUP_NAME = "default_group"

# 1. 特征列定义类 (Feature Columns) 

class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    """
    稀疏特征定义（类别特征）。
    """
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        # 如果未指定 embedding_name，默认与 feature_name 相同
        # 如果多个特征共享同一个 Embedding 矩阵（如源域用户ID和目标域用户ID），可以将它们的 embedding_name 设为相同
        if embedding_name is None:
            embedding_name = name
        # 自动计算 Embedding 维度：6 * (vocab_size ^ 0.25)，一种经验公式
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print(
                "Notice! Feature Hashing on the fly currently is not supported in torch version,you can use tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    """
    变长稀疏特征定义（序列特征）。
    例如：用户历史点击的商品 ID 序列。
    通常需要配合 Pooling (mean/sum/max) 将变长序列处理为定长向量。
    """
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    # 代理属性访问到内部的 SparseFeat
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    """
    稠密特征定义（数值特征）。
    不需要 Embedding，直接输入全连接层。
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


# 2. 输入构建逻辑 (Input Building) 

def build_input_features(feature_columns):
    """
    核心函数：构建特征名到输入索引范围的映射。
    
    DeepCTR-Torch 的 forward 输入通常是一个巨大的 Tensor (Batch_Size, Total_Features)。
    这个函数计算每个特征在这个大 Tensor 中占据的列索引范围 (start, end)。
    
    Return:
        OrderedDict: {feature_name: (start_index, end_index)}
    """
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        # 稀疏特征占 1 列 (存储类别 ID)
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        # 稠密特征占 dimension 列 (通常为 1)
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        # 变长特征占 maxlen 列 (存储序列 ID，不足的通常补0)
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            # 如果指定了 length_name (记录实际序列长度的特征)，它也占 1 列
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """
    将稀疏特征的 Embedding 向量和稠密特征的数值拼接，作为 DNN 的输入。
    
    Args:
        sparse_embedding_list: list of [batch, 1, embed_dim]
        dense_value_list: list of [batch, 1]
    Return:
        Tensor: [batch, total_input_dim]
    """
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        # 展平并拼接稀疏 Embedding: [batch, num_sparse * embed_dim]
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        # 展平并拼接稠密数值: [batch, num_dense]
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        # 最终拼接
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    """
    处理变长序列特征的 Pooling 操作。
    """
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        # 获取序列中每个 ID 对应的 Embedding
        seq_emb = embedding_dict[feat.name]
        
        if feat.length_name is None:
            # 如果没有指定长度特征，假设 0 是 padding 值，生成 mask
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0
            
            # 使用 SequencePoolingLayer 进行聚合 (mean/sum/max)
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        else:
            # 如果指定了实际长度，使用长度进行截断/聚合
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False, device=device)(
                [seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


# 3. Embedding 管理与查找 (Embedding Management) 

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    """
    根据特征定义创建 Embedding 层。
    
    Args:
        linear: 是否为线性模型部分创建 Embedding (维度为1)。
    Return:
        nn.ModuleDict: {embedding_name: nn.Embedding}
    """
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    # 使用 nn.ModuleDict 存储，key 是 embedding_name
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # 权重初始化
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


def embedding_lookup(X, sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    """
    Embedding 查找核心函数。
    
    Args:
        X: 输入大 Tensor [batch_size, total_features]
        sparse_embedding_dict: {embedding_name: nn.Embedding}
        sparse_input_dict: {feature_name: (start_idx, end_idx)}
        sparse_feature_columns: 特征定义列表
        
    Return:
        group_embedding_dict: defaultdict(list)，按 group 分组后的 Embedding 列表。
        对于大多数模型，所有特征都在 'default_group'。
    """
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            # 1. 找到该特征在输入 Tensor X 中的列位置
            lookup_idx = np.array(sparse_input_dict[feature_name])
            # 2. 切片取出 ID 数据，并转为 LongTensor
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            # 3. 查表
            emb = sparse_embedding_dict[embedding_name](input_tensor)
            # 4. 存入结果字典
            group_embedding_dict[fc.group_name].append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(X, embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    """
    变长特征的 Embedding 查找。
    返回字典 {feature_name: embedding_tensor}
    注意：这里查出来的 Tensor 形状是 [batch, maxlen, embed_dim]，还没做 Pooling。
    """
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            # Hash 处理逻辑 (To-Do)
            lookup_idx = sequence_input_dict[feature_name]
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](
            X[:, lookup_idx[0]:lookup_idx[1]].long())

    return varlen_embedding_vec_dict


def get_dense_input(X, features, feature_columns):
    """
    提取稠密特征数值。
    """
    dense_feature_columns = list(filter(lambda x: isinstance(
        x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = np.array(features[fc.name])
        # 切片并转 float
        input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].float()
        dense_input_list.append(input_tensor)
    return dense_input_list


def maxlen_lookup(X, sparse_input_dict, maxlen_column):
    """
    查找变长序列的最大长度 (如果最大长度也是作为一个输入特征传入的话)。
    """
    if maxlen_column is None or len(maxlen_column)==0:
        raise ValueError('please add max length column for VarLenSparseFeat of DIN/DIEN input')
    lookup_idx = np.array(sparse_input_dict[maxlen_column[0]])
    return X[:, lookup_idx[0]:lookup_idx[1]].long()