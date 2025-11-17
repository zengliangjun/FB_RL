import torch
from typing import Dict

from fbutils.eval_mode import eval_mode

from base import agents
from cpr.agents import configs
from cpr.models import fb

class FBAgent(agents.BaseAgent):

    calcute: fb.FBCalcute
    target_calcute: fb.FBCalcute

    def __init__(self, cfg: configs.FBAgentConfig):
        super(FBAgent, self).__init__(cfg)
        self.calcute = self.model.calcute
        self.target_calcute = self.model.target_calcute

    @torch.no_grad()
    def _encode_expert(self, state: torch.Tensor):
        # encode expert trajectories through B
        B = self.calcute.backward_representation(state).detach()  # batch x d
        B = B.view(- 1, self.config.seq_length, B.shape[-1])  # N x L x d
        z = B.mean(dim=1)  # N x d
        z = self.calcute.project_z(z)
        z = torch.repeat_interleave(z, self.config.seq_length, dim=0)  # batch x d
        return z

    @torch.no_grad()
    def _sample_mixed_z(self, state: torch.Tensor, expert_z: torch.Tensor):
        z = self.calcute.sample_z(self.config.batch_size)

        p_goal = self.config.sample_goal_ratio
        p_expert = self.config.expert_sample_ratio

        prob = torch.tensor(
            [p_goal, p_expert, 1 - p_goal - p_expert], dtype=torch.float32, device=z.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=self.config.batch_size, replacement=True).reshape(-1, 1)

        # zs obtained by encoding train goals
        perm = torch.randperm(self.config.batch_size, device=z.device)
        goals = self.calcute.backward_representation(state[perm])

        z = torch.where(mix_idxs == 0, goals, z)

        # zs obtained by encoding expert trajectories
        perm = torch.randperm(self.config.batch_size, device=z.device)
        z = torch.where(mix_idxs == 1, expert_z[perm], z)

        if hasattr(self, "z_buffer") and self.z_buffer is not None:
            self.z_buffer.extend(z)
        return z


    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_buffer = replay_buffer["expert_slicer"]
        rollout_buffer = replay_buffer["train"]

        expert_batch = expert_buffer.sample(self.config.batch_size)
        rollout_batch = rollout_buffer.sample(self.config.batch_size)

        ##
        expert_obs, expert_next_obs = (
            expert_batch["observation"].to(self.model.config.device),
            expert_batch["next"]["observation"].to(self.model.config.device),
        )

        ##
        rollout_obs, rollout_action, rollout_z, rollout_next_obs, rollout_terminated = (
            rollout_batch["observation"].to(self.model.config.device),
            rollout_batch["action"].to(self.model.config.device),
            rollout_batch["z"].to(self.model.config.device),
            rollout_batch["next"]["observation"].to(self.model.config.device),
            rollout_batch["next"]["terminated"].to(self.model.config.device),
        )
        rollout_discount = self.config.discount * ~rollout_terminated

        ##
        self.model.state_prepose(rollout_obs)
        self.model.state_prepose(rollout_next_obs)

        with torch.no_grad(), eval_mode(self.model.state_prepose):
            expert_obs, expert_next_obs = self.model.state_prepose(expert_obs), self.model.state_prepose(expert_next_obs)
            rollout_obs, rollout_next_obs = self.model.state_prepose(rollout_obs), self.model.state_prepose(rollout_next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        ## discriminator
        expert_z = self._encode_expert(state=expert_next_obs)

        input = {
            'expert_state': expert_obs,
            'expert_z_policy': expert_z,
            'state': rollout_obs,
            'z_policy': rollout_z,

        }
        discriminator_metrics = self.discriminator_updater.update(input, step)

        if self.config.relabel_ratio is not None:
            ## sample z
            z = self._sample_mixed_z(state=rollout_next_obs, expert_z=expert_z)

            mask = torch.rand((self.config.batch_size, 1), device=z.device) <= self.config.relabel_ratio
            rollout_z = torch.where(mask, z, rollout_z)

        inputs = {
                'state': rollout_obs,
                'action': rollout_action,
                'z_policy': rollout_z,

                'discount': rollout_discount,
                'next_state': rollout_next_obs
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
        metrics.update(actor_metrics)

        return metrics
