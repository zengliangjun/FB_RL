import torch
import torch.nn as nn
from typing import List, Any
import dataclasses
from torch.nn import functional as F
from safetensors.torch import load_model as safetensors_load_model
import os.path as osp

import math
from loguru import logger as ulogger

from fbutils import actor_post, utils
from base import configs, networks

@dataclasses.dataclass
class BaseModelConfig(configs.BaseConfig):

    device: str = "cuda:0"
    state_dimension: int = 0
    action_dimension: int = 0
    z_dimension: int = 0

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 1e-4  # 网络的学习率

    target_tau: float = 0.005

    _network_names_: List[str] = dataclasses.field(default_factory=list)  # 网络字段名
    _target_network_names_: List[str] = dataclasses.field(default_factory=list)  # 目标网络字段名

    _calcute_class_name_: str = ""  # 目标网络字段名

    def __post_init__(self):
        super().__post_init__()

        copy_from_model_field_names = [
                       "device",
                       "state_dimension",
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

            cfg.__post_init__()

    def network_names(self):
        return self._network_names_

    def target_names(self):
        return self._target_network_names_

    def network_config(self, name):
        cfg = getattr(self, name)
        assert cfg is not None
        assert hasattr(cfg, "_target_")
        assert isinstance(cfg, networks.BaseNetConfig)
        return cfg

class ModelsProxy():

    def __new__(cls, cfg: BaseModelConfig):

        modules: nn.Module = cfg.instantiate_from_config()
        modules.to(cfg.device)
        return modules


class BaseModelCalcute():

    state_prepose: nn.Module

    forward_map: nn.Module
    backward_map: nn.Module
    actor: nn.Module
    critic: nn.Module
    discriminator: nn.Module

    def __init__(self, config: BaseModelConfig):
        self.config = config

    def next_act(self, inputs: dict, type: actor_post.ActorValueType):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]
        return self.act(state=next_state, z_policy=z_policy, type = type)

    def act(self, state: torch.Tensor, z_policy: torch.Tensor, type: actor_post.ActorValueType, *args, **kwargs):
        actor = self.actor(state, z_policy)
        return actor_post.actor_post_process(self.config.net_actor, actor, type)

    def act_sample(self, actor_dist):
        assert not isinstance(actor_dist, torch.Tensor)
        return actor_post.actor_sample(self.config.net_actor, actor_dist)

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor, z_policy: torch.Tensor, *args, **kwargs):
        if hasattr(self, "state_prepose") and self.state_prepose is not None:
            state = self.state_prepose(state)
        return self.act(state, z_policy, actor_post.ActorValueType.MEAN)

    def next_forward_representation(self, inputs: dict, action: torch.Tensor):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]
        return self.forward_representation(next_state, action, z_policy)

    def forward_representation(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        return self.forward_map(state = state, action = action, z_policy = z_policy)  # batch x z_dim

    def backward_representation(self, state: torch.Tensor):
        z = self.backward_map(state = state)  # batch x z_dim
        return self.project_z(z)

    def discriminator_reward(self, state: torch.Tensor, z_policy: torch.Tensor):
        s = self.discriminator(state, z_policy)
        s = torch.sigmoid(s)
        ## forward
        eps: float = 1e-7

        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward

    def discriminator_logits(self, state: torch.Tensor, z_policy: torch.Tensor):
        return self.discriminator(state, z_policy)

    ###
    def sample_z(self, size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = self.config.device

        z = torch.randn((size, self.config.z_dimension), dtype=torch.float32, device=device)

        return self.project_z(z)

    ###
    def project_z(self, z):
        if self.config.net_backward_map.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


class BaseModel(nn.Module):

    calcute: BaseModelCalcute
    target_calcute: BaseModelCalcute

    def __init__(self, cfg: BaseModelConfig):
        super(BaseModel, self).__init__()
        self.config = cfg

        self._init_calcute()
        self._init_network()
        self.init_target()

    def _rawname(self, name):
        rawname = name
        if name.startswith("net_"):
            rawname = name[4:]
        return rawname

    def _init_calcute(self):
        import importlib
        calcute_name = self.config._calcute_class_name_

        modules_name, class_name = calcute_name.split(":")
        modules = importlib.import_module(modules_name)
        assert hasattr(modules, class_name), f"Module {modules_name} has no attribute {class_name}"

        class_obj = getattr(modules, class_name)
        self.calcute = class_obj(self.config)
        self.target_calcute = class_obj(self.config)

    def _init_network(self):

        lr = self.config.optim_lr
        weight_decay = self.config.optim_weight_decay

        for name in self.config.network_names():
            net_cfg: networks.BaseNetConfig = self.config.network_config(name)
            network: nn.Module = networks.NetProxy(net_cfg)

            rawname = self._rawname(name)
            ##
            setattr(self, rawname, network)
            setattr(self.calcute, rawname, network)
            setattr(self.target_calcute, rawname, network)

            optim_lr = utils.getattr_fix(net_cfg, "optim_lr", lr)
            optim_weight_decay = utils.getattr_fix(net_cfg, "optim_weight_decay", weight_decay)

            optim = torch.optim.Adam(params = network.parameters(),
                                          lr = optim_lr,
                                          weight_decay=optim_weight_decay)

            setattr(self, f"{rawname}_optim", optim)

    def init_target(self):
        for name in self.config.target_names():

            source_name = self._rawname(name)
            target_name = f"target_{name}"

            need_paramlist_init = False

            if not hasattr(self, target_name):
                cfg = self.config.network_config(name)
                target_network = networks.NetProxy(cfg)
                setattr(self, target_name, target_network)

                setattr(self.target_calcute, source_name, target_network)

                need_paramlist_init = True

            else:
                target_network = getattr(self, target_name)

            source_network = getattr(self, source_name)
            target_network.load_state_dict(source_network.state_dict())

            if need_paramlist_init:
                paramlist = tuple(x for x in source_network.parameters())
                target_paramlist = tuple(x for x in target_network.parameters())

                name = f'{source_name}_paramlist'
                tname = f'{target_name}_paramlist'
                setattr(self, name, paramlist)
                setattr(self, tname, target_paramlist)


    @torch.no_grad()
    def update_params(self):

        _target_tau = self.config.target_tau

        for name in self.config.target_names():

            source_name = self._rawname(name)
            target_name = f"target_{name}"

            net_cfg = self.config.network_config(name)

            param_name = f'{source_name}_paramlist'
            param_target_name = f'{target_name}_paramlist'

            target_tau = utils.getattr_fix(net_cfg, "target_tau", _target_tau)

            paramlist = getattr(self, param_name)
            target_paramlist = getattr(self, param_target_name)

            _soft_update_params(
                paramlist,
                target_paramlist,
                target_tau,
            )

    def _collect_names(self):
        net_names = []
        for name in self.config.network_names():

            rawname = self._rawname(name)
            net_names.append(rawname)

        for name in self.config.target_names():

            rawname = self._rawname(name)
            target_name = f"target_{rawname}"
            net_names.append(target_name)

        return net_names

    def save_dict(self, collect_dict: dict):
        net_names = self._collect_names()
        for name in net_names:
            net: nn.Module = getattr(self, name)
            collect_dict[name] = net.state_dict()

    def resume_dict(self, collect_dict: dict):
        net_names = self._collect_names()
        for name in net_names:
            net: nn.Module = getattr(self, name)
            net.load_state_dict(collect_dict[name])

    def resume(self, checkpoints_folder):
        model_file = osp.join(checkpoints_folder, "model.safetensors")

        ## model
        if osp.exists(model_file):
            safetensors_load_model(self, model_file, device=self.config.device)
            self.init_target()
            ulogger.info(f"resume model: {model_file}")



def _soft_update_params(net_params: Any, target_net_params: Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)

def soft_update_params(net, target_net, tau) -> None:
    tau = float(min(max(tau, 0), 1))
    net_params = tuple(x.data for x in net.parameters())
    target_net_params = tuple(x.data for x in target_net.parameters())
    _soft_update_params(net_params, target_net_params, tau)
