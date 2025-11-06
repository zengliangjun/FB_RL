import math

import torch
from torch import nn
import torch.nn.functional as F

from cpr.networks import configs
from cpr.networks import utils

class ResidualForwardMap(nn.Module):
    """残差前向表示网络

    基于残差连接的前向表示网络，通过跳跃连接提高深层网络的训练稳定性。
    支持并行计算用于不确定性估计。
    """
    def __init__(self, cfg: configs.ResidualForwardMapConfig, out_dims = -1) -> None:
        super().__init__()
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

        # 构建残差嵌入网络
        # 状态-技能向量残差嵌入网络（支持并行）
        self.embed_z = utils.residual_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers, num_parallel)
        # 状态-动作残差嵌入网络（支持并行）
        self.embed_sa = utils.residual_embedding(state_dimension + action_dimension, hidden_dimension, embedding_layers, num_parallel)

        # 构建残差F网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.ResidualBlock(hidden_dimension, num_parallel) for _ in range(hidden_layers)]

        if -1 == out_dims:
            out_dims = z_dimension
        seq += [utils.Block(hidden_dimension, out_dims, num_parallel)]  # 输出层

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

        # 合并嵌入向量并通过残差F网络
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))



class ResidualActor(nn.Module):
    """残差网络策略网络

    基于残差连接的策略网络，通过跳跃连接缓解深度网络的梯度消失问题。
    适用于较深的网络结构。
    """
    def __init__(self, cfg: configs.ResidualActorConfig) -> None:
        super(ResidualActor, self).__init__()
        assert not cfg.boltzmann, "残差网络不支持Boltzmann策略"

        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension = cfg.state_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建残差嵌入网络
        # 状态-技能向量残差嵌入网络
        self.embed_z = utils.residual_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers)
        # 状态残差嵌入网络
        self.embed_s = utils.residual_embedding(state_dimension, hidden_dimension, embedding_layers)

        # 构建残差策略网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.ResidualBlock(hidden_dimension) for _ in range(hidden_layers)] + \
              [utils.Block(hidden_dimension, out_dims, False)]  # 输出层不使用激活函数
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
        z_embedding = self.embed_z(torch.cat([state, z_policy], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(state)  # bs x h_dim // 2

        # 合并嵌入向量
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)

        # 通过残差策略网络生成动作
        return self.policy(embedding)


class ResidualCritic(ResidualForwardMap):
    """前向表示网络

    学习状态-动作对到下一个状态技能向量的映射。
    在强化学习中用于预测给定状态和动作后的技能向量变化。
    """
    def __init__(self, cfg: configs.ResidualCriticConfig) -> None:
        super(ResidualCritic, self).__init__(cfg)
