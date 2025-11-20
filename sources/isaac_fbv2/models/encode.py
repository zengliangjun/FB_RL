
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Union
import math


from base import models
from cpr.networks import utils
from isaac_fbv2.models import configs

from fbutils import actor_post

class SimBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps = 0.00001, momentum = 0.1, affine = True, track_running_stats = True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        reshape = False
        if 3 == len(input.shape):
            reshape = True
            b, _, dim = input.shape
            input = torch.reshape(input, (-1, dim))

        input = super().forward(input)

        if reshape:
            input = torch.reshape(input, (b, -1, dim))

        return input


class Calcute():

    privileges_prepose: nn.Module
    state_prepose: nn.Module

    critic_embed: nn.Module

    forward_map: nn.Module
    critic: nn.Module
    reward_critic: nn.Module

    actor_vae: nn.Module
    actor: nn.Module

    discriminator: nn.Module
    backward_map: nn.Module

    def __init__(self, cfg: configs.ModelConfig):
        self.config = cfg

    def next_act(self, inputs: dict, type: actor_post.ActorValueType):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]

        observations, _ = state
        next_observations, next_privileges = next_state

        next_observations = torch.cat((observations[:, 1:, :], next_observations[:, None, :]), dim = 1)

        next_state = next_observations, next_privileges        
        return self.act(state=next_state, z_policy=z_policy, type = type)

    def act(self, state: torch.Tensor, z_policy: torch.Tensor, type: actor_post.ActorValueType, *args, **kwargs):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            state = state[0]
            assert 3 == len(state.shape)

        if 3 == len(state.shape):
            state = torch.reshape(state, (state.shape[0], -1))

        if hasattr(self, "actor_vae"):
            latent = self.actor_vae.sample(state)
            actor = self.actor(latent, z_policy)
        else:
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

    def _build_next_state(self, inputs: dict):
        state = inputs["state"]
        next_state = inputs["next_state"]
        observations, _ = state
        next_observations, next_privileges = next_state

        next_observations = torch.cat((observations[:, 1:, :], next_observations[:, None, :]), dim = 1)

        next_state = next_observations, next_privileges
        return next_state

    def next_forward_representation(self, inputs: dict, action: torch.Tensor):
        z_policy = inputs["z_policy"]
        next_state = self._build_next_state(inputs)
        return self.forward_representation(next_state, action, z_policy)

    def forward_representation(self, state: Tuple[torch.Tensor], action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        embed = self.critic_embed(state = state, action = action, z_policy = z_policy)
        return self.forward_map(embed)  # batch x z_dim

    def next_critic_calcute(self, inputs: dict, action: torch.Tensor):
        z_policy = inputs["z_policy"]
        next_state = self._build_next_state(inputs)
        return self.critic_calcute(next_state, action, z_policy)

    def critic_calcute(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        embed = self.critic_embed(state = state, action = action, z_policy = z_policy)
        return self.critic(embed)

    def next_reward_calcute(self, inputs: dict, action: torch.Tensor):
        z_policy = inputs["z_policy"]
        next_state = self._build_next_state(inputs)
        return self.reward_calcute(next_state, action, z_policy)

    def reward_calcute(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        embed = self.critic_embed(state = state, action = action, z_policy = z_policy)
        return self.reward_critic(embed)

    def backward_representation(self, state: Tuple[torch.Tensor]):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            assert len(state[0].shape) == 2
            state = torch.cat(state, dim = -1)

        z = self.backward_map(state = state)  # batch x z_dim
        return self.project_z(z)

    def discriminator_mergerstate(self, state: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            if len(state[0].shape) == 3:
                observations, privileges = state
                observations = observations[:, -1]
                state = (observations, privileges)

            state = torch.cat(state, dim = -1)
        return state

    def discriminator_reward(self, state: Tuple[torch.Tensor], z_policy: torch.Tensor):
        state = self.discriminator_mergerstate(state)

        s = self.discriminator(state, z_policy)
        s = torch.sigmoid(s)
        ## forward
        eps: float = 1e-7

        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward

    def discriminator_logits(self, state: torch.Tensor, z_policy: torch.Tensor):
        state = self.discriminator_mergerstate(state)
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

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> torch.Tensor:
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

class  Model(models.BaseModel):
    config: configs.ModelConfig

    def __init__(self, cfg: configs.ModelConfig):
        super(Model, self).__init__(cfg)
        print("Model: \n",  self)

    def _init_network(self):
        super(Model, self)._init_network()

        self.private_prepose = nn.BatchNorm1d(self.config.privileges_dimension, affine=False, momentum=0.01)
        setattr(self.calcute, "private_prepose", self.private_prepose)
        setattr(self.target_calcute, "private_prepose", self.private_prepose)

        self.state_prepose = SimBatchNorm1d(self.config.state_dimension, affine=False, momentum=0.01)
        setattr(self.calcute, "state_prepose", self.state_prepose)
        setattr(self.target_calcute, "state_prepose", self.state_prepose)

        self.apply(utils.weight_init)
