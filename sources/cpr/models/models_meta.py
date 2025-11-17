import sys
import importlib
from typing import List
from torch import nn
import dataclasses
import copy
import math
import torch
from torch.nn import functional as F
import numpy as np

from base import configs, networks, models
from fbutils import actor_post, utils

metamotivo_dir = "/workspace/data2/VSCODE/MOTION/MATA/metamotivo"

if metamotivo_dir not in sys.path:
    sys.path.insert(0, metamotivo_dir)

from metamotivo.fb  import model  as fbmodel
from metamotivo.fb_cpr import model  as cprmodel



@dataclasses.dataclass
class MetaFBConfig(configs.BaseConfig):
    _target_: str = "metamotivo.fb.model:FBModel"

    metaconfig: fbmodel.Config = dataclasses.field(default_factory=fbmodel.Config)

    device: str = "cuda:0"

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 1e-4  # 网络的学习率

    target_tau: float = 0.005

    _network_names_: List[str] = dataclasses.field(default_factory=list)  # 网络字段名
    _target_network_names_: List[str] = dataclasses.field(default_factory=list)  # 目标网络字段名

    _calcute_class_name_: str = "cpr.models.models_meta:MetaCalcute"  # 目标网络字段名

    def network_names(self):
        return self._network_names_

    def target_names(self):
        return self._target_network_names_

    def network_config(self, name):
        cfg = getattr(self, name)
        return cfg



@dataclasses.dataclass
class MetaCPRConfig(MetaFBConfig):
    _target_: str = "metamotivo.fb_cpr.model:FBcprModel"

    metaconfig: cprmodel.Config = dataclasses.field(default_factory=cprmodel.Config)



def instantiate_from_config(cfg: MetaFBConfig) -> object:
    """Convert an object to a file path based on its module and class name."""
    modules_name, class_name = cfg._target_.split(":")
    modules = importlib.import_module(modules_name)
    assert hasattr(modules, class_name), f"Module {modules_name} has no attribute {class_name}"

    class_obj = getattr(modules, class_name)
    return class_obj(**dataclasses.asdict(cfg.metaconfig))


class ModelsProxy():

    def __new__(cls, cfg: MetaFBConfig):
        meta_modules: nn.Module = instantiate_from_config(cfg)
        meta_modules.to(cfg.metaconfig.device)
        return BaseModel(meta_modules, cfg)

class BaseModelCalcute():
    def __init__(self, cfg: MetaFBConfig):
        self.config = cfg


class BaseModel(nn.Module):

    calcute: BaseModelCalcute
    target_calcute: BaseModelCalcute

    def __init__(self, model: nn.Module, cfg: MetaFBConfig):
        super(BaseModel, self).__init__()
        self.config = cfg

        # '_obs_normalizer',
        for name in ['_backward_map', '_forward_map', '_actor',
                     '_target_backward_map', '_target_forward_map',
                     '_discriminator', '_critic',
                     '_target_critic']:

            if hasattr(model, name):
                _submodel = getattr(model, name)
                setattr(self, name[1: ], copy.deepcopy(_submodel))


        self._init_calcute()
        self._init_network()
        self.init_target()

        _submodel = getattr(model, "_obs_normalizer")
        setattr(self, "state_prepose", _submodel)
        setattr(self.calcute, "state_prepose", _submodel)
        setattr(self.target_calcute, "state_prepose", _submodel)

        self.train(True)
        self.requires_grad_(True)


    def _init_calcute(self):
        import importlib
        calcute_name = self.config._calcute_class_name_

        modules_name, class_name = calcute_name.split(":")
        modules = importlib.import_module(modules_name)
        assert hasattr(modules, class_name), f"Module {modules_name} has no attribute {class_name}"

        class_obj = getattr(modules, class_name)
        self.calcute = class_obj(self.config)
        self.target_calcute = class_obj(self.config)

    @staticmethod
    def _rawname(name):
        rawname = name
        if name.startswith("net_"):
            rawname = name[4:]
        return rawname

    def _init_network(self):

        lr = self.config.optim_lr
        weight_decay = self.config.optim_weight_decay

        for name in self.config.network_names():
            net_cfg: networks.BaseNetConfig = self.config.network_config(name)

            rawname = BaseModel._rawname(name)

            network: nn.Module = getattr(self, rawname)

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

            source_name = BaseModel._rawname(name)
            target_name = f"target_{source_name}"

            need_paramlist_init = False

            source_network = getattr(self, source_name)
            if not hasattr(self, target_name):
                target_network = copy.deepcopy(source_network)
                setattr(self, target_name, target_network)
                setattr(self.target_calcute, source_name, target_network)

                need_paramlist_init = True

            else:
                target_network = getattr(self, target_name)

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
            target_name = f"target_{source_name}"

            net_cfg = self.config.network_config(name)

            param_name = f'{source_name}_paramlist'
            param_target_name = f'{target_name}_paramlist'

            target_tau = utils.getattr_fix(net_cfg, "target_tau", _target_tau)

            paramlist = getattr(self, param_name)
            target_paramlist = getattr(self, param_target_name)

            models._soft_update_params(
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


class MetaCalcute(BaseModelCalcute):

    state_prepose: nn.Module

    forward_map: nn.Module
    backward_map: nn.Module
    actor: nn.Module
    critic: nn.Module
    discriminator: nn.Module

    def __init__(self, cfg: MetaFBConfig):
        super(MetaCalcute, self).__init__(cfg)


    def act(self, state: torch.Tensor, z_policy: torch.Tensor, type: actor_post.ActorValueType, *args, **kwargs):
        dist = self.actor(state, z_policy, self.config.net_actor.std)
        if type == actor_post.ActorValueType.MEAN:
            return dist.mean
        elif type == actor_post.ActorValueType.SAMPLE:

            return dist.sample()
        elif type == actor_post.ActorValueType.DISTRIBUTION:
            return dist

    def act_sample(self, actor_dist):
        return actor_dist.sample()

    @torch.no_grad()
    def act_inference(self, state: torch.Tensor, z_policy: torch.Tensor, *args, **kwargs):
        state = self.state_prepose(state)
        dist = self.actor(state, z_policy, self.config.net_actor.std)
        return dist.mean

    def forward_representation(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        return self.forward_map(state, z_policy, action)

    def backward_representation(self, state: torch.Tensor):
        return self.backward_map(state)

    def discriminator_reward(self, state: torch.Tensor, z_policy: torch.Tensor):
        return self.discriminator.compute_reward(obs=state, z=z_policy)

    def discriminator_logits(self, state: torch.Tensor, z_policy: torch.Tensor):
        return self.discriminator.compute_logits(state, z_policy)

    ###
    def sample_z(self, size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = self.config.device

        z = torch.randn((size, self.config.metaconfig.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    ###
    def project_z(self, z):
        if self.config.net_backward_map.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> torch.Tensor:
        """计算目标值和不确定性

        基于并行网络的预测结果计算均值、不确定性和悲观估计。
        用于离线强化学习中的不确定性惩罚。

        Args:
            preds: 并行网络的预测结果，形状为 [num_parallel, ...]
            pessimism_penalty: 悲观惩罚系数

        Returns:
            preds_mean: 预测均值
            preds_unc: 预测不确定性
            pessimistic_target: 悲观目标值（均值 - 惩罚 * 不确定性）
        """
        dim = 0  # 并行维度
        preds_mean = preds.mean(dim=dim)  # 计算均值

        # 计算所有网络对之间的差异
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...

        # 计算不确定性（所有网络对差异的平均值）
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]  # 网络对数量
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),  # 在并行维度上求和
            )
            / num_parallel_scaling  # 除以网络对数量得到平均值
        )

        # 计算悲观目标值
        # preds_mean, preds_unc,
        return preds_mean - pessimism_penalty * preds_unc


    def reward_inference(self, state: torch.Tensor, \
                         reward: torch.Tensor, \
                         inference_batch_size: int = 500_000,
                         weight: torch.Tensor | None = None) -> torch.Tensor:

        state = self.state_prepose(state)

        num_batches = int(np.ceil(state.shape[0] / inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * inference_batch_size, (i + 1) * inference_batch_size


            B = self.backward_representation(state[start_idx:end_idx].to(self.config.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.config.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, state: torch.Tensor,
                                  reward: torch.Tensor,
                                  inference_batch_size: int = 500_000) -> torch.Tensor:

        state = self.state_prepose(state)

        return self.reward_inference(state = state,
                                     reward = reward,
                                     inference_batch_size = inference_batch_size,
                                    weight = F.softmax(10 * reward, dim=0))

    def goal_inference(self, state: torch.Tensor) -> torch.Tensor:
        state = self.state_prepose(state)

        return self.backward_representation(state.to(self.config.device))

    def tracking_inference(self, state: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "seq_length")

        state = self.state_prepose(state)

        z = self.backward_representation(state.to(self.config.device))
        for step in range(z.shape[0]):
            end_idx = min(step + self.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
