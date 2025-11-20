import dataclasses
from typing import Optional

from base import networks

@dataclasses.dataclass
class CriticEmbedConfig(networks.BaseNetConfig):
    _target_: str = "isaac_fbv2.networks.residual.encode:CriticEmbed"  # 目标类路径

    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 2
    hidden_layers: int = 2

    target_tau: float = 0.01

    def __post_init__(self):
        super(CriticEmbedConfig, self).__post_init__()

@dataclasses.dataclass
class CriticBlockConfig(CriticEmbedConfig):
    _target_: str = "isaac_fbv2.networks.residual.encode:CriticBlock"  # 目标类路径

    def __post_init__(self):
        super(CriticBlockConfig, self).__post_init__()
        self.out_dimension = 1


@dataclasses.dataclass
class ForwardMapBlockConfig(CriticEmbedConfig):
    _target_: str = "isaac_fbv2.networks.residual.encode:CriticBlock"  # 目标类路径

    def __post_init__(self):
        super(ForwardMapBlockConfig, self).__post_init__()
        self.out_dimension = self.z_dimension


@dataclasses.dataclass
class StateVAEConfig(networks.BaseNetConfig):
    _target_: str = "isaac_fbv2.networks.residual.encode:StateVAE"  # 目标类路径

    hidden_dimension: int = 1024
    embedding_layers: int = 2
    hidden_layers: int = 2

    latent_dimension: int = 1024



@dataclasses.dataclass
class ActorConfig(networks.ActorConfig):
    _target_: str = "isaac_fbv2.networks.residual.encode:ResidualActor"  # 目标类路径

    hidden_dimension: int = 1024
    embedding_layers: int = 2
    hidden_layers: int = 2

    latent_dimension: int = 1024
