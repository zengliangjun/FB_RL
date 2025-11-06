from base import updaters

import dataclasses


@dataclasses.dataclass
class FBConfig(updaters.BaseUpdaterConfig):
    _target_: str = "cpr.updaters.fbupdater:FBUpdater"  # 目标类路径

    if_calcute_q: bool = False
    q_loss_coef: float = 0

    pessimism_penalty: float = 0
    ortho_coef: float = 1
    clip_grad_norm: float = 0


@dataclasses.dataclass
class ActorConfig(updaters.BaseUpdaterConfig):
    _target_: str = "cpr.updaters.actorupdater:ActorUpdater"  # 目标类路径

    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0

