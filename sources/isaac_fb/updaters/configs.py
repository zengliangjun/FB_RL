import dataclasses

from base import updaters
from cpr.updaters.cpr import configs as cpr_configs


@dataclasses.dataclass
class ActorConfig(cpr_configs.ActorConfig):
    _target_: str = "isaac_fb.updaters.actor:ActorUpdater"  # 目标类路径

    scale_reg: bool = False
    reg_coeff: float = 0.05
    reword_coeff: float = 0.02

@dataclasses.dataclass
class RewardCriticConfig(updaters.BaseUpdaterConfig):
    _target_: str = "isaac_fb.updaters.reward_critic:RewardCriticUpdater"  # 目标类路径

    pessimism_penalty: float = 0.5
