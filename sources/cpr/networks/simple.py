import math

import torch
from torch import nn
import torch.nn.functional as F

from cpr.networks import configs
from cpr.networks import utils

##########################
# 前向表示网络
##########################

class SimpleForwardMap(nn.Module):
    """简单前向表示网络

    基于传统MLP的前向表示网络，学习状态-动作对到技能向量的映射。
    支持并行计算用于不确定性估计。
    """
    def __init__(self, cfg: configs.SimpleForwardMapConfig) -> None:
        super(SimpleForwardMap, self).__init__()
        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension: int = cfg.state_dimension  # 状态维度
        action_dimension: int = cfg.action_dimension  # 动作维度
        z_dimension: int = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度
        embedding_layers = cfg.embedding_layers  # 嵌入层数量

        num_parallel = cfg.num_parallel  # 并行网络数量
        hidden_layers = cfg.hidden_layers  # F网络层数

        # 构建嵌入网络
        # 状态-技能向量嵌入网络（支持并行）
        self.embed_z = utils.simple_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers, num_parallel)
        # 状态-动作嵌入网络（支持并行）
        self.embed_sa = utils.simple_embedding(state_dimension + action_dimension, hidden_dimension, embedding_layers, num_parallel)

        # 构建F网络（前向表示网络）
        seq = []
        for _ in range(hidden_layers):
            seq += [utils.linear(hidden_dimension, hidden_dimension, num_parallel), nn.ReLU()]  # 线性层 + ReLU

        seq += [utils.linear(hidden_dimension, out_dims, num_parallel)]  # 输出层
        self.Fs = nn.Sequential(*seq)

    def forward(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        """前向传播

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]
            action: 动作张量，形状为 [batch_size, action_dimension]
            z_policy: 技能向量，形状为 [batch_size, z_dimension]

        Returns:
            前向表示输出，形状为 [num_parallel, batch_size, z_dimension]
        """
        num_parallel = self.config.num_parallel

        # 如果需要并行计算，扩展输入维度
        if num_parallel > 1:
            state = state.expand(num_parallel, -1, -1)  # [num_parallel, batch_size, state_dim]
            z_policy = z_policy.expand(num_parallel, -1, -1)  # [num_parallel, batch_size, z_dim]
            action = action.expand(num_parallel, -1, -1)  # [num_parallel, batch_size, action_dim]

        # 分别嵌入状态-技能向量和状态-动作
        z_embedding = self.embed_z(torch.cat([state, z_policy], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([state, action], dim=-1))  # num_parallel x bs x h_dim // 2

        # 合并嵌入向量并通过F网络
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))



class BackwardMap(nn.Module):
    """后向表示网络

    学习状态到技能向量的映射。
    在强化学习中用于将状态映射到对应的技能表示空间。
    使用简单的MLP架构，没有残差连接。
    """
    def __init__(self, cfg: configs.SimpleBackwardMapConfig) -> None:
        super(BackwardMap, self).__init__()
        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension: int = cfg.state_dimension  # 状态维度
        z_dimension: int = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # z_dimension

        hidden_dimension: int = cfg.hidden_dimension  # 隐藏层维度
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建后向表示网络
        seq = [nn.Linear(state_dimension, hidden_dimension), nn.LayerNorm(hidden_dimension), nn.Tanh()]  # 输入层
        for _ in range(hidden_layers-1):
            seq += [nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()]  # 隐藏层
        seq += [nn.Linear(hidden_dimension, out_dims)]  # 输出层

        self.B = nn.Sequential(*seq)

    def forward(self, state: torch.Tensor):
        """前向传播

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]

        Returns:
            技能向量表示，形状为 [batch_size, z_dimension]
        """
        B = self.B(state)  # 通过后向网络
        return B



class SimpleActor(nn.Module):
    """简单MLP策略网络

    基于传统多层感知机的策略网络，没有残差连接。
    适用于较浅的网络结构。
    """
    def __init__(self, cfg: configs.SimpleActorConfig) -> None:
        super(SimpleActor, self).__init__()
        assert not cfg.boltzmann, "简单网络不支持Boltzmann策略"

        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension = cfg.state_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建嵌入网络
        # 状态-技能向量嵌入网络
        self.embed_z = utils.simple_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers)
        # 状态嵌入网络
        self.embed_s = utils.simple_embedding(state_dimension, hidden_dimension, embedding_layers)

        # 构建策略网络
        seq = []
        for _ in range(hidden_layers):
            seq += [utils.linear(hidden_dimension, hidden_dimension), nn.ReLU()]  # 线性层 + ReLU激活
        seq += [utils.linear(hidden_dimension, out_dims)]  # 输出层
        self.policy = nn.Sequential(*seq)

    def forward(self,
        state: torch.Tensor,
        z_policy: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]
            z_policy: 技能向量，形状为 [batch_size, z_dimension]

        Returns:
            动作张量，形状为 [batch_size, action_dimension]
        """
        # 分别嵌入状态-技能向量和状态
        z_embedding = self.embed_z(torch.cat([state, z_policy], dim = -1))  # bs x h_dim // 2
        s_embedding = self.embed_s(state)  # bs x h_dim // 2

        # 合并嵌入向量
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)

        # 通过策略网络生成动作
        return self.policy(embedding)



class SimpleCritic(SimpleForwardMap):
    """前向表示网络

    学习状态-动作对到下一个状态技能向量的映射。
    在强化学习中用于预测给定状态和动作后的技能向量变化。
    """
    def __init__(self, cfg: configs.SimpleCriticConfig) -> None:
        super(SimpleCritic, self).__init__(cfg)


class Discriminator(nn.Module):
    def __init__(self, cfg: configs.SimpleDiscriminatorConfig) -> None:
        super(Discriminator, self).__init__()

        self.config = cfg  # 配置对象

        state_dimension: int = cfg.state_dimension
        z_dimension: int = cfg.z_dimension

        hidden_dimension: int = cfg.hidden_dimension
        hidden_layers: int = cfg.hidden_layers


        seq = [nn.Linear(state_dimension + z_dimension, hidden_dimension), nn.LayerNorm(hidden_dimension), nn.Tanh()]
        for _ in range(hidden_layers-1):
            seq += [nn.Linear(hidden_dimension, hidden_dimension), nn.ReLU()]
        seq += [nn.Linear(hidden_dimension, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, state: torch.Tensor, z_policy: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, z_policy], dim=1)
        return self.trunk(x)
