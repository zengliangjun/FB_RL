import torch.nn as nn
import dataclasses
from typing import Optional
from base import configs


@dataclasses.dataclass
class BaseNetConfig(configs.BaseConfig):

    device: Optional[str] = None

    state_dimension: Optional[int] = None  # 状态空间维度
    action_dimension: Optional[int] = None  # 动作空间维度
    z_dimension: Optional[int] = None  # 技能向量维度

    out_dimension: Optional[int] = None  # 技能向量维度

    optim_lr: Optional[float] = None #1e-4  # 网络的学习率
    optim_weight_decay: Optional[float] = None #1e-4  # 网络的学习率

class NetProxy():

    def __new__(self, cfg: BaseNetConfig):
        return cfg.instantiate_from_config()

####################################################################################
#
#
#
#
#
####################################################################################

@dataclasses.dataclass
class FConfig(BaseNetConfig):

    target_tau: Optional[float] = None

    def __post_init__(self):
        super(FConfig, self).__post_init__()
        self.out_dimension = self.z_dimension

@dataclasses.dataclass
class BConfig(BaseNetConfig):
    """后向表示网络配置类

    配置后向表示网络的参数，后向表示学习状态到技能向量的映射。
    在强化学习中用于将状态编码为对应的技能表示。
    """
    target_tau: Optional[float] = None

    norm_z: bool = True  # 是否对输出进行归一化

    def __post_init__(self):
        super(BConfig, self).__post_init__()
        self.out_dimension = self.z_dimension

        delattr(self, "action_dimension")

@dataclasses.dataclass
class ActorConfig(BaseNetConfig):
    # 策略采样方式
    boltzmann: bool = False  # 是否使用Boltzmann策略
    # Boltzmann策略相关配置
    log_std_bounds: tuple = (-5.0, 2.0)  # 对数标准差边界，用于约束探索程度
    # 非Boltzmann策略配置
    #stddev_schedule: str = "0.2"  # 标准差调度策略，如 "linear(1,0.2,200000)"
    std: float = 0.2  # 标准差
    stddev_clip: float = 0.3  # 标准差裁剪值，控制探索噪声范围

    def __post_init__(self):
        super(ActorConfig, self).__post_init__()
        self.out_dimension = self.action_dimension


@dataclasses.dataclass
class CriticConfig(BaseNetConfig):

    target_tau: Optional[float] = None

    def __post_init__(self):
        super(CriticConfig, self).__post_init__()
        self.out_dimension = 1


@dataclasses.dataclass
class DiscriminatorConfig(BaseNetConfig):

    def __post_init__(self):
        super(DiscriminatorConfig, self).__post_init__()
        self.out_dimension = 1
        delattr(self, "action_dimension")
