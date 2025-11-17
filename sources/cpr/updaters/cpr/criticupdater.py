from base import updaters, models

from cpr.updaters.cpr import configs
from cpr.models import fb_cpr
from fbutils.actor_post import ActorValueType

from typing import Dict, Union
import torch
from torch.nn import functional as F
from torch.optim import Optimizer


class CriticUpdater(updaters.Updater):

    config: configs.CriticConfig

    calcute: fb_cpr.FBCprCalcute
    target_calcute: fb_cpr.FBCprCalcute

    critic_optim: Optimizer

    def __init__(self, cfg: configs.CriticConfig, model: models.BaseModel):
        super(CriticUpdater, self).__init__(cfg, model)

        self.calcute = model.calcute
        self.target_calcute = model.target_calcute

        self.critic_optim = getattr(model, "critic_optim")

    @torch.no_grad()
    def _calcute_target_Q(self, inputs: dict, step: int) -> dict:

        reward = self.target_calcute.discriminator_reward(state=inputs["state"], z_policy=inputs["z_policy"])

        next_action = self.target_calcute.act(state=inputs["next_state"], z_policy=inputs["z_policy"], type = ActorValueType.SAMPLE)
        next_Qs = self.target_calcute.critic(state=inputs["next_state"], action = next_action, z_policy=inputs["z_policy"])  # num_parallel x batch x 1

        discount = inputs["discount"]
        if isinstance(next_Qs, Union[tuple, list]) and len(next_Qs) == 2:
            next_Qs = torch.min(*next_Qs)
            target_Q = reward + discount * next_Qs
        else:
            if len(next_Qs.shape) == 3:
                # num_parallel = next_Qs.shape[0]
                next_Qs = self.target_calcute.get_targets_uncertainty(next_Qs, self.config.pessimism_penalty)
                target_Q = reward + discount * next_Qs
                # target_Q = target_Q.expand(num_parallel, -1, -1)
            else:
                target_Q = reward + discount * next_Qs

        return target_Q, {
                "critic/target_Q": target_Q.mean().detach(),
                "critic/disc_reward": reward.mean().detach()
            }

    def _calcute_loss(self, inputs: dict, step: int):
        target_Q, metrics = self._calcute_target_Q(inputs, step)
        # compute critic loss
        Qs = self.calcute.critic(state=inputs["state"], action = inputs["action"], z_policy=inputs["z_policy"])  # num_parallel x batch x (1 or n_bins)

        if isinstance(Qs, Union[tuple, list]) and len(Qs) == 2:
            critic_loss = 0.5 * sum(F.mse_loss(Qsi, target_Q) for Qsi in Qs)
        else:
            if len(Qs.shape) == 3:
                num_parallel = Qs.shape[0]
                target_Q = target_Q.expand(num_parallel, -1, -1)
                critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, target_Q)
            else:
                critic_loss = 0.5 * F.mse_loss(Qs, target_Q)

        with torch.no_grad():
            output_metrics = {
                "critic/critic_Q": Qs.mean().detach(),
                "critic/loss": critic_loss.detach()
            }
            output_metrics.update(metrics)

        return critic_loss, output_metrics

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        # 优化FB网络
        self.critic_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), self.config.clip_grad_norm
            )

        self.critic_optim.step()

        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_critic_optim"] = self.critic_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.critic_optim.load_state_dict(collect_dict[f"{prefix}_critic_optim"])
