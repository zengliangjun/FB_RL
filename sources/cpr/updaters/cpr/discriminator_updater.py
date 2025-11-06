from base import updaters, models

from cpr.updaters.cpr import configs
from cpr.models import fb_cpr
from utils.actor_post import ActorValueType

from typing import Dict, Union
import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch import autograd


class DiscriminatorUpdater(updaters.Updater):

    config: configs.DiscriminatorConfig

    calcute: fb_cpr.FBCprCalcute
    target_calcute: fb_cpr.FBCprCalcute

    discriminator_optim: Optimizer

    def __init__(self, cfg: configs.DiscriminatorConfig, model: models.BaseModel):
        super(DiscriminatorUpdater, self).__init__(cfg, model)

        self.calcute = model.calcute
        self.target_calcute = model.target_calcute

        self.discriminator_optim = getattr(model, "discriminator_optim")


    @torch.compiler.disable
    def _gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor,
        real_z: torch.Tensor,
        fake_obs: torch.Tensor,
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=real_obs.device)
        interpolates = torch.cat(
            [
                (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
                (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True),
            ],
            dim=1,
        )
        d_interpolates = self.calcute.discriminator_logits(
            interpolates[:, 0 : real_obs.shape[1]], interpolates[:, real_obs.shape[1] :]
        )
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _calcute_loss(self, batch: dict, step: int) -> dict:
        expert_state, expert_z =  batch['expert_state'], batch['expert_z_policy']
        state, z = batch['state'], batch['z_policy']

        expert_logits = self.calcute.discriminator_logits(state=expert_state, z_policy=expert_z)
        unlabeled_logits = self.calcute.discriminator_logits(state=state, z_policy=z)
        # these are equivalent to binary cross entropy
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        if self.config.grad_penalty is not None:
            wgan_gp = self._gradient_penalty_wgan(expert_state, expert_z, state, z)
            loss += self.config.grad_penalty * wgan_gp

        with torch.no_grad():
            output_metrics = {
                "disc/loss": loss.detach(),
                "disc/expert_loss": expert_loss.detach().mean(),
                "disc/train_loss": unlabeled_loss.detach().mean()
            }
            if self.config.grad_penalty is not None:
                output_metrics["disc/wgan_gp_loss"] = wgan_gp.detach()
        return loss, output_metrics

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        loss, metrics = self._calcute_loss(batch, step)
        # 优化FB网络
        self.discriminator_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.discriminator.parameters(), self.config.clip_grad_norm
            )

        self.discriminator_optim.step()

        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_discriminator_optim"] = self.discriminator_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.discriminator_optim.load_state_dict(collect_dict[f"{prefix}_discriminator_optim"])
