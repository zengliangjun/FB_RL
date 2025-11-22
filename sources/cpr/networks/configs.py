from base import networks

import dataclasses

@dataclasses.dataclass
class SimpleForwardMapConfig(networks.FConfig):
    _target_: str = "cpr.networks.simple:SimpleForwardMap"  # 目标类路径

    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 2
    hidden_layers: int = 2

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    target_tau: float = 0.01


@dataclasses.dataclass
class ResidualForwardMapConfig(SimpleForwardMapConfig):
    _target_: str = "cpr.networks.residual:ResidualForwardMap"  # 目标类路径

@dataclasses.dataclass
class BreezeForwardMapConfig(SimpleForwardMapConfig):
    _target_: str = "cpr.networks.breezes:Critic"  # 目标类路径

    embedding_simple: bool = True

@dataclasses.dataclass
class SimpleBackwardMapConfig(networks.BConfig):
    _target_: str = "cpr.networks.simple:BackwardMap"  # 目标类路径

    hidden_dimension: int = 256
    hidden_layers: int = 1

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    target_tau: float = 0.01


@dataclasses.dataclass
class BreezeBackwardMapConfig(SimpleBackwardMapConfig):
    _target_: str = "cpr.networks.breezes:Backward"  # 目标类路径

    d_model: int = 256
    nhead: int = 8
    dropout: float = 0.1

@dataclasses.dataclass
class SimpleActorConfig(networks.ActorConfig):
    _target_: str = "cpr.networks.simple:SimpleActor"  # 目标类路径

    hidden_dimension: int = 1024
    embedding_layers: int = 2
    hidden_layers: int = 2

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0


@dataclasses.dataclass
class ResidualActorConfig(SimpleActorConfig):
    _target_: str = "cpr.networks.residual:ResidualActor"  # 目标类路径


@dataclasses.dataclass
class SimpleCriticConfig(SimpleForwardMapConfig):
    _target_: str = "cpr.networks.simple:SimpleCritic"  # 目标类路径

    def __post_init__(self):
        super(SimpleCriticConfig, self).__post_init__()
        self.out_dimension = 1

@dataclasses.dataclass
class ResidualCriticConfig(ResidualForwardMapConfig):
    _target_: str = "cpr.networks.residual:ResidualCritic"  # 目标类路径

    def __post_init__(self):
        super(ResidualCriticConfig, self).__post_init__()
        self.out_dimension = 1

@dataclasses.dataclass
class SimpleDiscriminatorConfig(networks.DiscriminatorConfig):
    _target_: str = "cpr.networks.simple:Discriminator"  # 目标类路径

    hidden_dimension: int = 1024
    hidden_layers: int = 2

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

@dataclasses.dataclass
class ResidualDiscriminatorConfig(SimpleDiscriminatorConfig):
    _target_: str = "cpr.networks.residual:ResidualActor"  # 目标类路径


@dataclasses.dataclass
class SimpleVConfig(SimpleDiscriminatorConfig):
    pass

@dataclasses.dataclass
class ResidualVConfig(ResidualDiscriminatorConfig):
    pass
