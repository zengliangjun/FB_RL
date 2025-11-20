import dataclasses
from typing import List

from base import networks, models


@dataclasses.dataclass
class  ModelConfig(models.BaseModelConfig):
    _target_: str = "isaac_fbv2.models.encode:Model"
    _calcute_class_name_: str = "isaac_fbv2.models.encode:Calcute"  # 计算类名

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 1e-4  # 网络的学习率

    target_tau: float = 0.005

    history_horizon: int = 4
    privileges_dimension: int = 607
    latent_dimension: int = 607

    def _init_network_names(self):
        self._network_names_ = [
            "net_critic_embed",

            "net_forward_map",
            "net_critic",
            "net_reward_critic",

            "net_actor_vae",
            "net_actor",

            "net_discriminator",
            "net_backward_map",
        ]
        self._target_network_names_ = [
            "net_critic_embed",

            "net_forward_map",
            "net_critic",
            "net_reward_critic",
            
            "net_backward_map",
        ]


    def __post_init__(self):
        self._init_network_names()

        super(ModelConfig, self).__post_init__()

        critic_embed_cfg: networks.BaseNetConfig = getattr(self, "net_critic_embed")
        setattr(critic_embed_cfg, "state_dimension", self.history_horizon * self.state_dimension + self.privileges_dimension)
        critic_embed_cfg.__post_init__()

        for name in ["net_forward_map", "net_critic", "net_reward_critic",]:
            cfg: networks.BaseNetConfig = getattr(self, name)
            setattr(cfg, "state_dimension", critic_embed_cfg.out_dimension)
            cfg.__post_init__()

        if hasattr(self, "net_actor_vae"):
            actor_vae_cfg: networks.BaseNetConfig = getattr(self, "net_actor_vae")
            setattr(actor_vae_cfg, "state_dimension", self.history_horizon * self.state_dimension)
            setattr(actor_vae_cfg, "out_dimension", self.privileges_dimension)
            actor_vae_cfg.__post_init__()

            actor_cfg: networks.BaseNetConfig = getattr(self, "net_actor")
            setattr(actor_cfg, "state_dimension", actor_vae_cfg.latent_dimension)
            actor_cfg.__post_init__()
        else:
            actor_cfg: networks.BaseNetConfig = getattr(self, "net_actor")
            setattr(actor_cfg, "state_dimension", self.history_horizon * self.state_dimension)
            actor_cfg.__post_init__()


        for name in ["net_backward_map", "net_discriminator"]:
            cfg: networks.BaseNetConfig = getattr(self, name)
            setattr(cfg, "state_dimension", self.state_dimension + self.privileges_dimension)
            cfg.__post_init__()

