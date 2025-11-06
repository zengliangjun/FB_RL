from base import models

import dataclasses
from typing import List


@dataclasses.dataclass
class  FBModelConfig(models.BaseModelConfig):
    _target_: str = "cpr.models.fb:FBModel"

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 1e-4  # 网络的学习率

    target_tau: float = 0.005

    _network_names_: List[str] = dataclasses.field(default_factory=list) # 网络字段名
    _target_network_names_: List[str] = dataclasses.field(default_factory=list) # 目标网络字段名

    _calcute_class_name_: str = "cpr.models.fb:FBCalcute"  # 计算类名

    def _init_network_names(self):
        self._network_names_ = ["net_forward_map", "net_backward_map", "net_actor"]
        self._target_network_names_ = ["net_forward_map", "net_backward_map"]

    def __post_init__(self):
        self._init_network_names()

        super().__post_init__()


@dataclasses.dataclass
class  FBCprModelConfig(FBModelConfig):
    _target_: str = "cpr.models.fb_cpr:FBCprModel"

    _calcute_class_name_: str = "cpr.models.fb_cpr:FBCprCalcute"  # 计算类名

    def _init_network_names(self):
        self._network_names_ = ["net_forward_map", "net_backward_map", "net_actor", "net_critic", "net_discriminator"]
        self._target_network_names_ = ["net_forward_map", "net_backward_map", "net_critic"]

