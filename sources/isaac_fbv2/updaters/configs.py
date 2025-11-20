import dataclasses

from base import updaters
from cpr.updaters import configs as fbconfigs
from cpr.updaters.cpr import configs as cprconfigs
from isaac_fb.updaters import configs as isaac_configs

@dataclasses.dataclass
class FBConfig(fbconfigs.FBConfig):
    _target_: str = "isaac_fbv2.updaters.fb_critic:FBUpdater"  # 目标类路径


@dataclasses.dataclass
class CriticConfig(cprconfigs.CriticConfig):
    _target_: str = "isaac_fbv2.updaters.critic:CriticUpdater"  # 目标类路径

@dataclasses.dataclass
class RewardCriticConfig(isaac_configs.RewardCriticConfig):
    _target_: str = "isaac_fbv2.updaters.reward_critic:RewardCriticUpdater"  # 目标类路径


@dataclasses.dataclass
class DiscriminatorConfig(cprconfigs.DiscriminatorConfig):
    _target_: str = "isaac_fbv2.updaters.discriminator:DiscriminatorUpdater"  # 目标类路径


@dataclasses.dataclass
class ActorConfig(isaac_configs.ActorConfig):
    _target_: str = "isaac_fbv2.updaters.actor:ActorUpdater"  # 目标类路径

@dataclasses.dataclass
class VAEConfig(updaters.BaseUpdaterConfig):
    _target_: str = "isaac_fbv2.updaters.state:VAEUpdater"  # 目标类路径
    kl_loss_coef: float = 0
    clip_grad_norm: float = 0
