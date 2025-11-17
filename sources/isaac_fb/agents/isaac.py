from  cpr.agents.fb_cpr_agents import FBAgent

import torch
from typing import Dict

from fbutils.eval_mode import eval_mode

from base import agents
from isaac_fb.agents import configs
from isaac_fb.models import isaac

class Agent(FBAgent):

    model: isaac.IsaacModel
    calcute: isaac.IsaacCalcute
    target_calcute: isaac.IsaacCalcute

    def __init__(self, cfg: configs.IsaacAgentConfig):
        super(FBAgent, self).__init__(cfg)
        self.calcute = self.model.calcute
        self.target_calcute = self.model.target_calcute

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_buffer = replay_buffer["expert_slicer"]
        rollout_buffer = replay_buffer["train"]

        expert_batch = expert_buffer.sample(self.config.batch_size)
        rollout_batch = rollout_buffer.sample(self.config.batch_size)

        ##
        expert_obs, expert_privileges, \
        expert_next_obs, expert_next_privileges  = (
            expert_batch["observations"].to(self.model.config.device),
            expert_batch["privileges"].to(self.model.config.device),
            expert_batch["next"]["observations"].to(self.model.config.device),
            expert_batch["next"]["privileges"].to(self.model.config.device),
        )

        ##
        rollout_obs, rollout_privileges, rollout_action, rollout_z, \
            rollout_next_obs, rollout_next_privileges, rollout_terminated, \
            rollout_rewards = (

            rollout_batch["observations"].to(self.model.config.device),
            rollout_batch["privileges"].to(self.model.config.device),
            rollout_batch["action"].to(self.model.config.device),
            rollout_batch["z"].to(self.model.config.device),
            rollout_batch["next"]["observations"].to(self.model.config.device),
            rollout_batch["next"]["privileges"].to(self.model.config.device),
            rollout_batch["next"]["terminated"].to(self.model.config.device),
            rollout_batch["next"]["rewards"].to(self.model.config.device),
        )
        rollout_discount = self.config.discount * ~rollout_terminated

        ##
        self.model.state_prepose(rollout_obs)
        self.model.state_prepose(rollout_next_obs)

        self.model.private_prepose(rollout_privileges)
        self.model.private_prepose(rollout_next_privileges)

        with torch.no_grad(), eval_mode(self.model.state_prepose):
            expert_obs = self.model.state_prepose(expert_obs)
            expert_next_obs = self.model.state_prepose(expert_next_obs)
            ##
            rollout_obs = self.model.state_prepose(rollout_obs)
            rollout_next_obs = self.model.state_prepose(rollout_next_obs)
            ##
            expert_privileges = self.model.private_prepose(expert_privileges)
            expert_next_privileges = self.model.private_prepose(expert_next_privileges)
            ##
            rollout_privileges = self.model.private_prepose(rollout_privileges)
            rollout_next_privileges = self.model.private_prepose(rollout_next_privileges)

        # torch.compiler.cudagraph_mark_step_begin()
        ## discriminator
        expert_z = self._encode_expert(state=[expert_next_obs, expert_next_privileges])

        input = {
            'expert_state': [expert_obs, expert_privileges],
            'expert_z_policy': expert_z,
            'state': [rollout_obs, rollout_privileges],
            'z_policy': rollout_z,
        }
        discriminator_metrics = self.discriminator_updater.update(input, step)

        inputs = {
                'state': [rollout_obs, rollout_privileges],
                'action': rollout_action,
                'z_policy': rollout_z,

                'reward': rollout_rewards,
                'discount': rollout_discount,
                'next_state': [rollout_next_obs, rollout_next_privileges],
            }
        reward_critic_metrics = self.reward_critic_updater.update(inputs, step)

        if self.config.relabel_ratio is not None:
            ## sample z
            next_state = torch.cat([rollout_next_obs, rollout_next_privileges], dim = -1)
            z = self._sample_mixed_z(state=next_state, expert_z=expert_z)

            mask = torch.rand((self.config.batch_size, 1), device=z.device) <= self.config.relabel_ratio
            rollout_z = torch.where(mask, z, rollout_z)

        inputs = {
                'state': [rollout_obs, rollout_privileges],
                'action': rollout_action,
                'z_policy': rollout_z,

                'reward': rollout_rewards,
                'discount': rollout_discount,
                'next_state': [rollout_next_obs, rollout_next_privileges],
            }

        fb_metrics = self.fb_updater.update(inputs, step)
        critic_metrics = self.critic_updater.update(inputs, step)
        actor_metrics = self.actor_updater.update(inputs, step)


        with torch.no_grad():
            self.model.update_params()

        metrics = {}

        metrics = {}
        metrics.update(discriminator_metrics)
        metrics.update(fb_metrics)
        metrics.update(critic_metrics)
        metrics.update(reward_critic_metrics)
        metrics.update(actor_metrics)

        return metrics
