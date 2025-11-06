"""
神经网络工具模块 - 概率分布相关

这个模块提供了强化学习中常用的概率分布类和变换函数。
包括Tanh变换、截断正态分布和压缩正态分布等，主要用于处理连续动作空间。

作者: Meta Platforms, Inc.
"""

import torch
from torch import distributions as pyd
import math
import torch.nn.functional as F


class TanhTransform(pyd.transforms.Transform):
    """Tanh变换类

    将实数空间映射到[-1, 1]区间的双射变换。
    在强化学习中常用于将无界动作值压缩到有界范围内。
    """
    domain = pyd.constraints.real  # 定义域：实数空间
    codomain = pyd.constraints.interval(-1.0, 1.0)  # 值域：[-1, 1]区间
    bijective = True  # 双射变换，确保可逆性
    sign = +1  # 变换的符号

    def __init__(self, cache_size=1) -> None:
        """初始化Tanh变换

        Args:
            cache_size: 缓存大小，用于提高性能
        """
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x) -> torch.Tensor:
        """反双曲正切函数

        计算x的反双曲正切值，即tanh的逆函数。

        Args:
            x: 输入张量，值域为[-1, 1]

        Returns:
            反双曲正切值，定义域为实数空间
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        """相等性判断

        Args:
            other: 另一个对象

        Returns:
            是否为相同类型的TanhTransform
        """
        return isinstance(other, TanhTransform)

    def _call(self, x) -> torch.Tensor:
        """前向变换

        将实数x通过tanh函数映射到[-1, 1]区间。

        Args:
            x: 输入张量，定义域为实数空间

        Returns:
            tanh(x)，值域为[-1, 1]
        """
        return x.tanh()

    def _inverse(self, y) -> torch.Tensor:
        """逆变换

        将[-1, 1]区间的值y通过atanh函数映射回实数空间。
        注意：不在边界处进行截断，以避免某些算法性能下降。

        Args:
            y: 输入张量，值域为[-1, 1]

        Returns:
            atanh(y)，定义域为实数空间
        """
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        """对数雅可比行列式的绝对值

        计算变换的对数雅可比行列式绝对值，用于概率密度变换。
        使用数值稳定的公式计算。

        Args:
            x: 变换前的值
            y: 变换后的值

        Returns:
            对数雅可比行列式的绝对值

        Reference:
            https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        """
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


from torch.distributions.utils import _standard_normal

class TruncatedNormal(pyd.Normal):
    """截断正态分布

    在标准正态分布基础上添加边界截断功能。
    在强化学习中用于生成有界动作，同时保持梯度传播。
    """
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        """初始化截断正态分布

        Args:
            loc: 均值参数
            scale: 标准差参数
            low: 下界，默认-1.0
            high: 上界，默认1.0
            eps: 边界容差，避免数值问题
        """
        super().__init__(loc, scale, validate_args=False)
        self.low = low  # 下界
        self.high = high  # 上界
        self.eps = eps  # 边界容差

    def _clamp(self, x) -> torch.Tensor:
        """截断函数

        将输入x截断到[low+eps, high-eps]范围内。
        使用梯度重定向技巧保持梯度传播。

        Args:
            x: 输入张量

        Returns:
            截断后的张量，前向传播使用截断值，反向传播使用原始梯度
        """
        # 将x截断到有效范围内
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        # 梯度重定向：前向传播使用clamped_x，反向传播梯度通过原始x传播
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        """采样函数

        从截断正态分布中采样，支持额外的clip参数控制采样范围。

        Args:
            clip: 额外的截断参数，控制噪声的范围
            sample_shape: 采样形状

        Returns:
            采样结果，形状为sample_shape
        """
        # 计算扩展后的形状
        shape = self._extended_shape(sample_shape)
        # 生成标准正态分布噪声
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale  # 应用标准差缩放

        # 如果提供了clip参数，对噪声进行额外截断
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)

        # 计算采样值：均值 + 噪声
        x = self.loc + eps
        # 应用边界截断
        return self._clamp(x)


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """压缩正态分布

    通过Tanh变换将正态分布压缩到[-1, 1]区间。
    在强化学习中常用于生成有界连续动作。
    """
    def __init__(self, loc, scale) -> None:
        """初始化压缩正态分布

        Args:
            loc: 均值参数
            scale: 标准差参数
        """
        self.loc = loc  # 均值
        self.scale = scale  # 标准差

        # 基础正态分布
        self.base_dist = pyd.Normal(loc, scale)
        # 变换序列：仅包含Tanh变换
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        """均值属性

        计算压缩后的分布均值。
        通过应用所有变换得到最终均值。

        Returns:
            压缩后的分布均值
        """
        mu = self.loc  # 原始均值
        for tr in self.transforms:
            mu = tr(mu)  # 应用变换
        return mu

import enum
from typing import Union

class ActorValueType(enum.Enum):
    MEAN = 0
    SAMPLE = 1
    DISTRIBUTION = 2


def actor_sample(config, dist: Union[SquashedNormal, TruncatedNormal]) -> torch.Tensor:

    if config.boltzmann:
        # Boltzmann策略：从SquashedNormal分布采样
        assert isinstance(dist, SquashedNormal)
        return dist.sample()
    else:
        # 确定性策略：从TruncatedNormal分布采样，应用标准差裁剪
        assert isinstance(dist, TruncatedNormal)
        return dist.sample(clip=config.stddev_clip)

def actor_post_process(config, actor: torch.Tensor, type: ActorValueType):
    if config.boltzmann:
        # Boltzmann策略：使用SquashedNormal分布
        mu, log_std = actor.chunk(2, dim=-1)  # 分割为均值和标准差

        # 约束对数标准差在指定范围内
        log_std = torch.tanh(log_std)  # 映射到[-1, 1]
        log_std_min, log_std_max = config.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()  # 转换为标准差
        dist = SquashedNormal(mu, std)  # 创建压缩正态分布
    else:
        # 确定性策略：使用TruncatedNormal分布
        std = config.std  # 根据步数调度标准差
        mu = torch.tanh(actor)  # 使用tanh约束动作范围到[-1, 1]
        std = torch.ones_like(mu) * std  # 扩展标准差到与均值相同形状
        dist = TruncatedNormal(mu, std)  # 创建截断正态分布

    if type == ActorValueType.MEAN:
        return dist.mean
    if type == ActorValueType.SAMPLE:
        return actor_sample(config, dist)

    return dist
