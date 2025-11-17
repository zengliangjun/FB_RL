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
    def _sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        z = self.calcute.sample_z(self.config.batch_size)

        if train_goal is not None:
            perm = torch.randperm(self.config.batch_size, device=z.device)
            goals = self.calcute.backward_representation(train_goal[perm])
            mask = torch.rand((self.config.batch_size, 1), device=z.device) < self.config.sample_goal_ratio
            z = torch.where(mask, goals, z)

        if hasattr(self, "z_buffer") and self.z_buffer is not None:
            self.z_buffer.extend(z)

        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        if isinstance(replay_buffer, dict):
            replay_buffer = replay_buffer["train"]
        batch = replay_buffer.sample(self.config.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        ##
        obs = obs.to(self.model.config.device)
        action = action.to(self.model.config.device)
        next_obs = next_obs.to(self.model.config.device)
        terminated = terminated.to(self.model.config.device)

        ##
        discount = self.config.discount * ~terminated

        ##
        self.model.state_prepose(obs)
        self.model.state_prepose(next_obs)

        with torch.no_grad(), eval_mode(self.model.state_prepose):
            obs, next_obs = self.model.state_prepose(obs), self.model.state_prepose(next_obs)

        #torch.compiler.cudagraph_mark_step_begin()
        mixed_z = self._sample_mixed_z(train_goal=next_obs)

        inputs = {
                'state': obs,
                'action': action,
                'z_policy': mixed_z,

                'discount': discount,
                'next_state': next_obs
            }


        #
        fb_metrics = self.fb_updater.update(inputs, step)
        actor_metrics = self.actor_updater.update(inputs, step)

        with torch.no_grad():
            self.model.update_params()

        metrics = {}
        metrics.update(fb_metrics)
        metrics.update(actor_metrics)

        return metrics
