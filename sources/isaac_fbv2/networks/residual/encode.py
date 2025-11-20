import torch
from torch import nn

from isaac_fbv2.networks.residual import configs
from cpr.networks import utils

class CriticEmbed(nn.Module):
    """残差前向表示网络

    基于残差连接的前向表示网络，通过跳跃连接提高深层网络的训练稳定性。
    支持并行计算用于不确定性估计。
    """
    def __init__(self, cfg: configs.CriticEmbedConfig) -> None:
        super().__init__()
        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension: int = cfg.state_dimension  # 状态维度
        action_dimension: int = cfg.action_dimension  # 动作维度
        z_dimension: int = cfg.z_dimension  # 技能向量维度

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度
        embedding_layers = cfg.embedding_layers  # 嵌入层数量
        hidden_layers = cfg.hidden_layers  # F网络层数

        num_parallel = cfg.num_parallel  # 并行网络数量

        # 构建残差嵌入网络
        # 状态-技能向量残差嵌入网络（支持并行）
        self.embed_z = utils.residual_embedding(state_dimension + z_dimension, hidden_dimension, embedding_layers, num_parallel)
        # 状态-动作残差嵌入网络（支持并行）
        self.embed_sa = utils.residual_embedding(state_dimension + action_dimension, hidden_dimension, embedding_layers, num_parallel)

        seq = [utils.ResidualBlock(hidden_dimension, num_parallel) for _ in range(hidden_layers)]
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
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))

class CriticBlock(nn.Module):
    """残差前向表示网络

    基于残差连接的前向表示网络，通过跳跃连接提高深层网络的训练稳定性。
    支持并行计算用于不确定性估计。
    """
    def __init__(self, cfg: configs.CriticBlockConfig) -> None:
        super().__init__()
        self.config = cfg  # 配置对象

        z_dimension: int = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        num_parallel = cfg.num_parallel  # 并行网络数量
        hidden_layers = cfg.hidden_layers  # F网络层数

        # 构建残差F网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.ResidualBlock(hidden_dimension, num_parallel) for _ in range(hidden_layers)]

        if -1 == out_dims:
            out_dims = z_dimension
        seq += [utils.Block(hidden_dimension, out_dims, num_parallel)]  # 输出层

        self.Fs = nn.Sequential(*seq)
    
    def forward(self, embedding: torch.Tensor):

        # 合并嵌入向量并通过残差F网络
        return self.Fs(embedding)

class StateEncoder(nn.Module):
    """残差网络策略网络
    https://github.com/LukeDitria/CNN-VAE/
    基于残差连接的策略网络，通过跳跃连接缓解深度网络的梯度消失问题。
    适用于较深的网络结构。
    """
    def __init__(self, cfg: configs.StateVAEConfig) -> None:
        super(StateEncoder, self).__init__()

        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension = cfg.state_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.latent_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建残差策略网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.Block(state_dimension, hidden_dimension, True)] + \
              [utils.ResidualBlock(hidden_dimension) for _ in range(hidden_layers)]
        self.embed = nn.Sequential(*seq)

        self.mu = utils.Block(hidden_dimension, out_dims, False)
        self.log_var = utils.Block(hidden_dimension, out_dims, False)


    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]
            z_policy: 技能向量，形状为 [batch_size, z_dimension]

        Returns:
            动作张量，形状为 [batch_size, action_dimension]
        """
        # 分别嵌入状态-技能向量和状态
        s_embedding = self.embed(state)

        mu = self.mu(s_embedding)  # 1
        log_var = self.log_var(s_embedding)  # 1

        if self.training:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var
    

class StateDecoder(nn.Module):
    """残差网络策略网络
    https://github.com/LukeDitria/CNN-VAE/
    基于残差连接的策略网络，通过跳跃连接缓解深度网络的梯度消失问题。
    适用于较深的网络结构。
    """
    def __init__(self, cfg: configs.StateVAEConfig) -> None:
        super(StateDecoder, self).__init__()
        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension = cfg.latent_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建残差策略网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.Block(state_dimension, hidden_dimension, True)] + \
              [utils.ResidualBlock(hidden_dimension) for _ in range(hidden_layers)]
        self.embed = nn.Sequential(*seq)

        self.dec = utils.Block(hidden_dimension, out_dims, False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]
            z_policy: 技能向量，形状为 [batch_size, z_dimension]

        Returns:
            动作张量，形状为 [batch_size, action_dimension]
        """
        # 分别嵌入状态-技能向量和状态
        s_embedding = self.embed(state)

        return self.dec(s_embedding)  # 1

class StateVAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, cfg: configs.StateVAEConfig):
        super(StateVAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        self.encoder = StateEncoder(cfg)
        self.decoder = StateDecoder(cfg)


    def encode(self, state: torch.Tensor):
        encoding, mu, log_var = self.encoder(state)
        return mu

    def sample(self, state: torch.Tensor):
        encoding, mu, log_var = self.encoder(state)
        return encoding

    def forward(self, state: torch.Tensor):
        encoding, mu, log_var = self.encoder(state)
        recon_state = self.decoder(encoding)
        return recon_state, mu, log_var



class ResidualActor(nn.Module):
    """残差网络策略网络

    基于残差连接的策略网络，通过跳跃连接缓解深度网络的梯度消失问题。
    适用于较深的网络结构。
    """
    def __init__(self, cfg: configs.ActorConfig) -> None:
        super(ResidualActor, self).__init__()
        assert not cfg.boltzmann, "残差网络不支持Boltzmann策略"

        self.config = cfg  # 配置对象

        # 获取维度参数
        state_dimension = cfg.latent_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量
        hidden_layers: int = cfg.hidden_layers  # 隐藏层数量

        # 构建残差策略网络
        # 使用残差块构建深层网络，最后接一个输出块
        seq = [utils.Block(state_dimension + z_dimension, hidden_dimension, True)] + \
              [utils.ResidualBlock(hidden_dimension) for _ in range(hidden_layers)] + \
              [utils.Block(hidden_dimension, out_dims, False)]  # 输出层不使用激活函数
        self.policy = nn.Sequential(*seq)

    def forward(self,
        state: torch.Tensor,
        z_policy: torch.Tensor) -> torch.Tensor:

        # 合并嵌入向量
        input = torch.cat([state, z_policy], dim=-1)

        # 通过残差策略网络生成动作
        return self.policy(input)
    