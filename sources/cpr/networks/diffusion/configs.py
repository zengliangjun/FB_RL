from base import configs, networks

import dataclasses
from typing import Optional

@dataclasses.dataclass
class IDQLConfig(configs.BaseConfig):
    _target_: str = "cpr.networks.diffusion.breezes:IDQLDiffusion"  # 目标类路径

    input_dimension: Optional[int] = None
    z_dimension: Optional[int] = None
    out_dimension: Optional[int] = None

    condition_dimension: int = 0
    time_dimension: int = 64
    time_embeding: str = 'fixed'
    action_fn: str = 'mish'

    hidden_dimension: int = 1024
    hidden_layers: int = 2


@dataclasses.dataclass
class ActorDiffusionConfig(networks.BaseNetConfig):
    _target_: str = "cpr.networks.diffusion.breezes:DiffusionActor"  # 目标类路径

    embedding_simple: bool = True

    hidden_dimension: int = 1024
    embedding_layers: int = 2


    diffusion_block: IDQLConfig = IDQLConfig()
    diffusion_schedule: str = 'cosine'
    diffusion_timesteps: int = 5

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    def __post_init__(self):
        super().__post_init__()

        self.diffusion_block.input_dimension = self.action_dimension
        self.diffusion_block.out_dimension = self.action_dimension
        self.diffusion_block.z_dimension = self.z_dimension
        self.diffusion_block.condition_dimension = self.hidden_dimension // 2

