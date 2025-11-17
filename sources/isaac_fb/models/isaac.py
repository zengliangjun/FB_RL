from torch import nn
import torch
from typing import Tuple, List, Union

from cpr.networks import utils
from cpr.models import fb_cpr
from isaac_fb.models import configs
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


class IsaacCalcute(fb_cpr.FBCprCalcute):

    private_prepose: nn.Module
    reward_critic: nn.Module

    def __init__(self, cfg: configs.IsaacConfig):
        super(IsaacCalcute, self).__init__(cfg)

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
        observations, _ = state
        next_observations, next_privileges = next_state

        next_observations = torch.cat((observations[:, 1:, :], next_observations[:, None, :]), dim = 1)

        next_state = next_observations, next_privileges
        return self.forward_representation(next_state, action, z_policy)

    def forward_representation(self, state: Tuple[torch.Tensor], action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        return self.forward_map(state = state, action = action, z_policy = z_policy)  # batch x z_dim

    def next_critic_calcute(self, inputs: dict, action: torch.Tensor):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]
        observations, _ = state
        next_observations, next_privileges = next_state

        next_observations = torch.cat((observations[:, 1:, :], next_observations[:, None, :]), dim = 1)

        next_state = next_observations, next_privileges
        return self.critic_calcute(next_state, action, z_policy)

    def critic_calcute(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        return self.critic(state, action, z_policy)

    def next_reward_calcute(self, inputs: dict, action: torch.Tensor):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]
        observations, _ = state
        next_observations, next_privileges = next_state

        next_observations = torch.cat((observations[:, 1:, :], next_observations[:, None, :]), dim = 1)

        next_state = next_observations, next_privileges
        return self.reward_calcute(next_state, action, z_policy)


    def reward_calcute(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            observations, privileges = state
            if 3 == len(observations.shape):
                observations = torch.reshape(observations, (observations.shape[0], -1))
            state = torch.cat((observations, privileges), dim = -1)

        return self.reward_critic(state, action, z_policy)

    def backward_representation(self, state: Tuple[torch.Tensor]):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            assert len(state[0].shape) == 2
            state = torch.cat(state, dim = -1)

        z = self.backward_map(state = state)  # batch x z_dim
        return self.project_z(z)

    def discriminator_reward(self, state: Tuple[torch.Tensor], z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            if len(state[0].shape) == 3:
                observations, privileges = state
                observations = observations[:, -1]
                state = (observations, privileges)
            state = torch.cat(state, dim = -1)

        s = self.discriminator(state, z_policy)
        s = torch.sigmoid(s)
        ## forward
        eps: float = 1e-7

        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward

    def discriminator_logits(self, state: torch.Tensor, z_policy: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            if len(state[0].shape) == 3:
                observations, privileges = state
                observations = observations[:, -1]
                state = (observations, privileges)

            state = torch.cat(state, dim = -1)

        return self.discriminator(state, z_policy)

    def discriminator_mergerstate(self, state: torch.Tensor):
        if isinstance(state, Union[tuple, list, Tuple, List]) and len(state) == 2:
            if len(state[0].shape) == 3:
                observations, privileges = state
                observations = observations[:, -1]
                state = (observations, privileges)

            state = torch.cat(state, dim = -1)
        return state



class  IsaacModel(fb_cpr.FBCprModel):
    config: configs.IsaacConfig

    private_prepose: nn.Module

    def __init__(self, cfg: configs.IsaacConfig):
        super(IsaacModel, self).__init__(cfg)
        print("IsaacModel: \n",  self)

    def _init_network(self):
        super(IsaacModel, self)._init_network()
        if not hasattr(self, "private_prepose"):
            self.private_prepose = nn.BatchNorm1d(self.config.privileges_dimension, affine=False, momentum=0.01)

            setattr(self.calcute, "private_prepose", self.private_prepose)
            setattr(self.target_calcute, "private_prepose", self.private_prepose)

        self.state_prepose = SimBatchNorm1d(self.config.state_dimension, affine=False, momentum=0.01)

        setattr(self.calcute, "state_prepose", self.state_prepose)
        setattr(self.target_calcute, "state_prepose", self.state_prepose)

        self.apply(utils.weight_init)
