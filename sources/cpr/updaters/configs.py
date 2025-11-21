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


@dataclasses.dataclass
class FBConservativeConfig(FBConfig):
    _target_: str = "cpr.updaters.conservativefb:FBUpdater"  # 目标类路径

    conservative_total_action_samples: int = 12
    conservative_ood_action_weight: float = 0.25  # should be multiple of 0.25
    conservative_alpha: float = 0.01
    conservative_target_conservative_penalty: float = 50.0
    conservative_critic_learning_rate: float = 1e-4

    if_measure_conservative_penalty: bool = True
    if_value_conservative_penalty: bool = True
