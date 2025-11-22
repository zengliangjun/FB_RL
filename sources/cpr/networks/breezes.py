import torch
from torch import nn
import torch.nn.functional as F

from cpr.networks import configs
from cpr.networks import utils

class SelfAttention(nn.Module):
    def __init__(self, z_dim: int, num_parallel: int = 1):
        super(SelfAttention, self).__init__()
        self.query = utils.linear(z_dim, z_dim, num_parallel)
        self.key = utils.linear(z_dim, z_dim, num_parallel)
        self.value = utils.linear(z_dim, z_dim, num_parallel)
        self.z_dim = z_dim
        self.num_parallel = num_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        if self.num_parallel == 1:
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            output = torch.bmm(attention_weights, V)
        else:
            _b, _d = Q.shape[1:]
            Q = torch.reshape(Q, (-1, _d))
            K = torch.reshape(K, (-1, _d))
            V = torch.reshape(V, (-1, _d))
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            output = torch.bmm(attention_weights, V)
            output = torch.reshape(Q, (-1, _b, _d))
        return output

class Critic(nn.Module):
    """简单前向表示网络

    基于传统MLP的前向表示网络，学习状态-动作对到技能向量的映射。
    支持并行计算用于不确定性估计。
    """
    def __init__(self, cfg: configs.BreezeForwardMapConfig) -> None:
        super(Critic, self).__init__()
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

        if cfg.embedding_simple:
            self.embed_z = utils.simple_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers, num_parallel)
            self.embed_sa = utils.simple_embedding(state_dimension + action_dimension, hidden_dimension, embedding_layers, num_parallel)
        else:
            self.embed_z = utils.residual_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers, num_parallel)
            self.embed_sa = utils.residual_embedding(state_dimension + action_dimension, hidden_dimension, embedding_layers, num_parallel)

        # 构建F网络（前向表示网络）
        seq = []
        for _ in range(hidden_layers):
            seq += [utils.BreezeBlock(hidden_dimension // 2)]
        self.block = nn.Sequential(*seq)

        self.FS = nn.Sequential(
                        utils.linear(hidden_dimension, hidden_dimension, num_parallel),
                        utils.linear(hidden_dimension, out_dims, num_parallel)
                    )


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

        if num_parallel > 1:
            mbedding = torch.cat([sa_embedding.unsqueeze(2), z_embedding.unsqueeze(2)], dim=-2)
            _p, _b, _k, _d = mbedding.shape
            mbedding = torch.reshape(mbedding, (-1, _k, _d))
            out = self.block(mbedding)
            out = torch.reshape(out, (_p, _b, _k * _d))
        else:
            mbedding = torch.cat([sa_embedding.unsqueeze(1), z_embedding.unsqueeze(1)], dim=-2)
            _b, _k, _d = mbedding.shape
            out = self.block(mbedding)
            out = torch.reshape(out, (_b, _k * _d))

        return self.FS(out)


class Backward(nn.Module):

    def __init__(self, cfg: configs.BreezeBackwardMapConfig) -> None:
        super(Backward, self).__init__()
        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension: int = cfg.state_dimension  # 状态维度
        z_dimension: int = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # z_dimension

        hidden_dimension: int = cfg.hidden_dimension  # 隐藏层维度
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量


        d_model: int = cfg.d_model
        nhead: int = cfg.nhead
        dropout: float = cfg.dropout

        # Input projection
        self.input_proj = torch.nn.Linear(state_dimension, d_model)

        # Positional encodings for encoder and decoder
        self.pos_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )
        self.pos_decoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )

        # Full Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=hidden_layers,
            num_decoder_layers=hidden_layers,
            dim_feedforward=hidden_dimension,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = torch.nn.Linear(d_model, out_dims)

        # Learnable query for the decoder
        self.query_embed = torch.nn.Parameter(torch.randn(1, d_model))

    def forward(self, state: torch.Tensor, position_encoding: bool = False) -> torch.Tensor:
        """
        Takes observation and processes it through full transformer architecture.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
        Returns:
            z: embedded tensor of shape [batch_dim, z_dimension]
        """
        batch_size = state.shape[0]

        if position_encoding:
            # Create position encodings for encoder
            src_positions = torch.arange(state.shape[1], dtype=torch.float32)\
                .expand(batch_size, -1).unsqueeze(-1)
            src_pos_encoding = self.pos_encoder(src_positions)

            # Project and add positional encoding for encoder input
            memory = self.input_proj(state)
            memory = memory.unsqueeze(1)
            memory = memory + src_pos_encoding
        else:
            x = state.unsqueeze(1)
            memory = self.input_proj(x)

        # Create decoder query
        query = self.query_embed.expand(batch_size, -1, -1)

        # Create position encoding for decoder
        tgt_positions = torch.zeros(batch_size, 1, 1, dtype=torch.float32).to(query.device)
        tgt_pos_encoding = self.pos_decoder(tgt_positions)
        query = query + tgt_pos_encoding

        # Generate target mask for decoder
        tgt_mask = torch.zeros((1, 1), dtype=torch.float32).to(query.device)

        # Pass through transformer
        output = self.transformer(
            src=memory,
            tgt=query,
            tgt_mask=tgt_mask
        )

        # Project to output dimension
        z = self.output_proj(output.squeeze(1))
        return z



