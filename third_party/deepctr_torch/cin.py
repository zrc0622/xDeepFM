class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
    xDeepFM 中的核心模块：压缩交互网络。
    用于显式地学习高阶特征组合（Explicit High-order Interaction）。
    它的主要思想是逐层计算特征向量的哈达玛积（Hadamard product），并通过类似卷积的操作压缩特征图数量。

      Input shape
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.
        - 输入必须是所有特征的 Embedding 向量拼接后的结果。
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` 
        - 输出是经过 Sum Pooling 后的高阶特征向量。
        - featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **field_size** : Positive integer, number of feature groups. (特征域的数量，即 M)
        - **layer_size** : list of int. Feature maps in each layer. (每一层 CIN 的特征图数量，即 H_k)
        - **activation** : activation function name used on feature maps. (激活函数)
        - **split_half** : bool. (是否将每一层的输出特征图切分一半，一半用于输出，一半用于下一层输入)
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024,
                 device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size] # 记录每一层的特征图数量 (H_0, H_1, ...)
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        # 构建每一层的卷积层（实际上是全连接层的作用，用于压缩特征交互维度）
        for i, size in enumerate(self.layer_size):
            # CIN 的核心公式：X^k = Conv(X^{k-1} ⊗ X^0)
            # 输入通道数 = 上一层的特征图数量 (H_{k-1}) * 原始输入特征域数量 (M, 即 H_0)
            # 输出通道数 = 当前层设定的特征图数量 (size, 即 H_k)
            # kernel_size=1: 沿着 Embedding 维度滑动，但因为核大小是1，实际就是对每个 Embedding 维度的特征图矩阵做线性加权组合
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                # 如果 split_half=True，除了最后一层外，中间层的输出 H_k 必须是偶数，
                # 因为一半要流入下一层，一半要直接输出作为结果。
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                # 下一层的输入维度仅为当前层输出维度的一半
                self.field_nums.append(size // 2)
            else:
                # 否则，全部输出都流入下一层
                self.field_nums.append(size)

        #         for tensor in self.conv1ds:
        #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs):
        # inputs 形状: [batch_size, field_size (M), embedding_size (D)]
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1] # Embedding 维度 D
        
        # hidden_nn_layers 保存每一层的输出状态 X^k
        # 初始状态 X^0 就是 inputs
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # 1. 计算交互矩阵 (Interaction) 
            # hidden_nn_layers[-1] 是上一层的输出 X^{k-1}，形状: [batch, H_{k-1}, D]
            # hidden_nn_layers[0] 是原始输入 X^0，形状: [batch, M, D]
            # torch.einsum('bhd,bmd->bhmd') 计算外积（哈达玛积的推广）
            # 结果形状: [batch, H_{k-1}, M, D]
            # 这步对应论文公式中 Z^{k+1} 的生成
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            
            # 2. Reshape 准备卷积 
            # 将中间两个维度合并，准备作为 Conv1d 的 input channels
            # x.shape 变为: [batch, H_{k-1} * M, D]
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            
            # 3. 压缩 (Compression) 
            # 使用 1D 卷积进行压缩，将 H_{k-1} * M 个特征图映射为 layer_size 个特征图
            # x.shape 变为: [batch, H_k, D]
            x = self.conv1ds[i](x)

            # 4. 激活函数
            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            # 5. 分流 (Splitting) 
            if self.split_half:
                # 如果不是最后一层，将特征图切分为两半
                if i != len(self.layer_size) - 1:
                    # next_hidden: 用于计算下一层 X^{k+1}
                    # direct_connect: 直接作为最终输出的一部分
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    # 最后一层全部用于输出，不产生 next_hidden
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                # 如果不切分，所有特征图既作为输出，也作为下一层的输入
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        # 6. 结果聚合 (Pooling)
        # 将所有层的 direct_connect 拼接
        # result 形状: [batch, total_feature_maps, D]
        result = torch.cat(final_result, dim=1)
        
        # Sum Pooling: 沿着 Embedding 维度 (dim=-1) 求和
        # 得到最终输出 p+，形状: [batch, total_feature_maps]
        # 这对应论文中将矩阵行向量求和得到标量的步骤
        result = torch.sum(result, -1)

        return result