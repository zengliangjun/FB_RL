"""
残差网络工具模块

这个模块提供了残差网络实现所需的各种工具函数和辅助模块。
包括并行计算支持、残差网络组件、网络构建工具和初始化方法。

作者: Meta Platforms, Inc.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numbers
import numpy as np

##########################
# 初始化工具
##########################

# 并行层的正交初始化
def parallel_orthogonal_(tensor, gain=1):
    """并行层的正交初始化

    对并行网络层的权重进行正交初始化，确保不同并行网络之间的独立性。
    基于QR分解实现，确保权重的正交性。

    Args:
        tensor: 需要初始化的权重张量
        gain: 增益系数，用于调整初始化范围

    Returns:
        初始化后的张量

    Raises:
        ValueError: 当张量维度不支持时
    """
    if tensor.ndimension() == 2:
        # 如果是2D张量，使用标准的正交初始化
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")

    n_parallel = tensor.size(0)  # 并行网络数量
    rows = tensor.size(1)  # 行数
    cols = tensor.numel() // n_parallel // rows  # 列数

    # 创建随机初始化的张量
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()  # 转置以确保行数 >= 列数

        # 计算QR分解
        q, r = torch.linalg.qr(flat_tensor)
        # 根据论文使Q均匀分布
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()  # 转置回来
        qs.append(q)

    qs = torch.stack(qs, dim=0)  # 重新堆叠
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)  # 复制正交化后的权重
        tensor.mul_(gain)  # 应用增益
    return tensor

def weight_init(m):
    """权重初始化函数

    对不同类型的层应用适当的初始化方法。

    Args:
        m: 需要初始化的模块
    """
    if isinstance(m, nn.Linear):
        # 线性层使用正交初始化
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)  # 偏置初始化为0
    elif isinstance(m, DenseParallel):
        # 并行线性层使用并行正交初始化
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)  # 偏置初始化为0
    elif hasattr(m, "reset_parameters"):
        # 如果模块有自己的初始化方法，使用它
        m.reset_parameters()


##########################
# 辅助模块
##########################

class DenseParallel(nn.Module):
    """并行密集层

    支持多个线性层并行计算的模块。
    用于集成学习和不确定性估计。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.n_parallel = n_parallel  # 并行网络数量

        if n_parallel is None or (n_parallel == 1):
            # 单网络情况
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            # 多网络并行情况
            self.weight = nn.Parameter(
                torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError("并行层必须支持偏置")
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        """从模块列表加载权重

        用于将多个独立模块的权重加载到并行层中。

        Args:
            module_list: 模块列表，长度必须等于n_parallel
        """
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]  # 转置权重
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO 为什么这些层有自己的重置方案？
    def reset_parameters(self) -> None:
        """重置参数

        使用Kaiming均匀初始化权重，使用均匀初始化偏置。
        """
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """前向传播

        Args:
            input: 输入张量

        Returns:
            输出张量
        """
        if self.n_parallel is None or (self.n_parallel == 1):
            # 单网络前向传播
            return F.linear(input, self.weight, self.bias)
        else:
            # 并行网络前向传播
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        """额外表示信息

        Returns:
            模块的字符串表示
        """
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    """并行层归一化

    支持多个层归一化并行计算的模块。
    """
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape, ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel  # 并行网络数量
        self.normalized_shape = list(normalized_shape)  # 归一化形状
        self.eps = eps  # 数值稳定性常数
        self.elementwise_affine = elementwise_affine  # 是否使用可学习参数

        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                # 单网络情况
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                # 多网络并行情况
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """重置参数

        权重初始化为1，偏置初始化为0。
        """
        if self.elementwise_affine:
            nn.init.ones_(self.weight)  # 权重初始化为1
            nn.init.zeros_(self.bias)  # 偏置初始化为0

    def load_module_list_weights(self, module_list) -> None:
        """从模块列表加载权重

        Args:
            module_list: 模块列表，长度必须等于n_parallel
        """
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        """前向传播

        Args:
            input: 输入张量

        Returns:
            归一化后的输出张量
        """
        norm_input = F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias  # 应用可学习参数
        else:
            return norm_input

    def extra_repr(self) -> str:
        """额外表示信息

        Returns:
            模块的字符串表示
        """
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


def linear(input_dim, output_dim, num_parallel=1):
    """线性层工厂函数

    根据并行数量创建线性层或并行线性层。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        num_parallel: 并行数量

    Returns:
        线性层模块
    """
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)  # 并行线性层
    return nn.Linear(input_dim, output_dim)  # 标准线性层

def layernorm(input_dim, num_parallel=1):
    """层归一化工厂函数

    根据并行数量创建层归一化或并行层归一化。

    Args:
        input_dim: 输入维度
        num_parallel: 并行数量

    Returns:
        层归一化模块
    """
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)  # 并行层归一化
    return nn.LayerNorm(input_dim)  # 标准层归一化

##########################
# 简单MLP模型
##########################
def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    """简单嵌入网络构建函数

    构建传统的MLP嵌入网络，没有残差连接。

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        hidden_layers: 隐藏层数量
        num_parallel: 并行数量

    Returns:
        嵌入网络Sequential模块

    Raises:
        AssertionError: 当隐藏层数量不足时
    """
    assert hidden_layers >= 2, "must have at least 2 embedding layers"

    # 构建嵌入网络序列
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]  # 输入层
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]  # 隐藏层
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]  # 输出层（维度减半）

    return nn.Sequential(*seq)

##########################
# 残差模型
##########################

class ResidualBlock(nn.Module):
    """残差块

    经典的残差连接模块，通过跳跃连接缓解梯度消失问题。
    包含LayerNorm、线性层和Mish激活函数。
    """
    def __init__(self, dim, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(dim, num_parallel)  # 层归一化
        lin = linear(dim, dim, num_parallel)  # 线性层
        self.mlp = nn.Sequential(ln, lin, nn.Mish())  # 子网络

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量 = 输入 + 子网络(输入)
        """
        return x + self.mlp(x)  # 残差连接

class Block(nn.Module):
    """基础块

    包含LayerNorm、线性层和可选激活函数的基础模块。
    """
    def __init__(self, input_dim, output_dim, activation, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(input_dim, num_parallel)  # 层归一化
        lin = linear(input_dim, output_dim, num_parallel)  # 线性层
        seq = [ln, lin] + ([nn.Mish()] if activation else [])  # 可选激活函数
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        return self.mlp(x)

def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    """残差嵌入网络构建函数

    构建基于残差连接的嵌入网络。

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        hidden_layers: 隐藏层数量
        num_parallel: 并行数量

    Returns:
        残差嵌入网络Sequential模块

    Raises:
        AssertionError: 当隐藏层数量不足时
    """
    assert hidden_layers >= 2, "must have at least 2 embedding layers"

    # 构建残差嵌入网络序列
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]  # 输入块
    for _ in range(hidden_layers-2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]  # 残差块
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]  # 输出块（维度减半）

    return nn.Sequential(*seq)


if __name__ == "__main__":
    module = DenseParallel(
        1024,
        256,
        2)

    print(module)
