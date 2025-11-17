import dataclasses
from base import networks
from cpr.models import configs

@dataclasses.dataclass
class  IsaacConfig(configs.FBCprModelConfig):
    _target_: str = "isaac_fb.models.isaac:IsaacModel"

    _calcute_class_name_: str = "isaac_fb.models.isaac:IsaacCalcute"  # 计算类名

    privileges_dimension: int = 607
    history_horizon: int = 4

    def _init_network_names(self):
        self._network_names_ = [
            "net_forward_map",
            "net_backward_map",
            "net_actor",
            "net_critic",
            "net_reward_critic",
            "net_discriminator",
        ]
        self._target_network_names_ = [
            "net_forward_map",
            "net_backward_map",
            "net_critic",
            "net_reward_critic"
        ]


    def __post_init__(self):
        super(IsaacConfig, self).__post_init__()

        copy_from_model_field_names = [
                       "device",
                       "action_dimension",
                       "z_dimension",
                       ]

        for name in self._network_names_:
            cfg: networks.BaseNetConfig = getattr(self, name)

            target_fields = {field.name for field in dataclasses.fields(cfg)}
            for field in copy_from_model_field_names:
                ## is None
                if field in target_fields:
                    setattr(cfg, field, getattr(self, field))

        for name in ["net_forward_map", "net_critic", "net_reward_critic"]:
            cfg: networks.BaseNetConfig = getattr(self, name)
            setattr(cfg, "state_dimension", self.history_horizon * self.state_dimension + self.privileges_dimension)
            cfg.__post_init__()

        for name in ["net_actor"]:
            cfg: networks.BaseNetConfig = getattr(self, name)
            setattr(cfg, "state_dimension", self.history_horizon * self.state_dimension)
            cfg.__post_init__()

        for name in ["net_backward_map", "net_discriminator"]:
            cfg: networks.BaseNetConfig = getattr(self, name)
            setattr(cfg, "state_dimension", self.state_dimension + self.privileges_dimension)
            cfg.__post_init__()

