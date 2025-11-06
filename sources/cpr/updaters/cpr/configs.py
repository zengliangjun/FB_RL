from base import updaters
import dataclasses

from cpr.updaters import configs as cpr_configs


@dataclasses.dataclass
class ActorConfig(cpr_configs.ActorConfig):
    _target_: str = "cpr.updaters.cpr.actorupdater:ActorUpdater"  # 目标类路径

    scale_reg: bool = True
    reg_coeff: float = 0.01

    pessimism_penalty: float = 0.5


@dataclasses.dataclass
class CriticConfig(updaters.BaseUpdaterConfig):
    _target_: str = "cpr.updaters.cpr.criticupdater:CriticUpdater"  # 目标类路径

    pessimism_penalty: float = 0.5


@dataclasses.dataclass
class DiscriminatorConfig(updaters.BaseUpdaterConfig):
    _target_: str = "cpr.updaters.cpr.discriminator_updater:DiscriminatorUpdater"  # 目标类路径
    grad_penalty: float = 10.0