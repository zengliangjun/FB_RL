from base import updaters
from typing import Dict, Union
import torch
from torch.nn import functional as F

from isaac_fbv2.models import encode
from isaac_fbv2.updaters import configs
from torch.optim import Optimizer

from isaac_fbv2.models import encode

def calcute_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

class VAEUpdater(updaters.Updater):

    config: configs.VAEConfig

    calcute: encode.Calcute
    target_calcute: encode.Calcute

    actor_vae_optim: Optimizer
    model: encode.Model
    def __init__(self, cfg: configs.VAEConfig, model: encode.Model):
        super(VAEUpdater, self).__init__(cfg, model)

        self.calcute = model.calcute
        self.target_calcute = model.target_calcute

        self.actor_vae_optim = getattr(model, "actor_vae_optim")

    def _calcute_loss(self, inputs: dict, step: int) -> dict:
        state = inputs["state"]
        rollout_obs, rollout_privileges = state

        rollout_obs = torch.reshape(rollout_obs, (rollout_obs.shape[0], -1))
        recon_privilege, mu, log_var = self.calcute.actor_vae(rollout_obs)

        kl_loss = calcute_kl_loss(mu, log_var)
        mse_loss = F.mse_loss(recon_privilege, rollout_privileges)

        loss = kl_loss * self.config.kl_loss_coef + mse_loss

        with torch.no_grad():
            metrics = {
                "vae/kl": kl_loss.detach(),
                "vae/mse": mse_loss.detach(),
                "vae/loss": loss.detach()
            }
        return loss, metrics

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        self.actor_vae_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.actor_vae.parameters(), self.config.clip_grad_norm
            )

        self.actor_vae_optim.step()
        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_actor_vae_optim"] = self.actor_vae_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.actor_vae_optim.load_state_dict(collect_dict[f"{prefix}_actor_vae_optim"])

