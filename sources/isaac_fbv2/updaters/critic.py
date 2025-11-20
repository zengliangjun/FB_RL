
from cpr.updaters.cpr import criticupdater
from isaac_fbv2.models import encode
from isaac_fbv2.updaters import configs

from torch.optim import Optimizer
import torch
from typing import Dict

class CriticUpdater(criticupdater.CriticUpdater):

    def __init__(self, cfg: configs.CriticConfig, model: encode.Model):
        super(CriticUpdater, self).__init__(cfg, model)
        self.critic_embed_optim = getattr(model, "critic_embed_optim")

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        # 优化FB网络
        self.critic_embed_optim.zero_grad(set_to_none=True)
        self.critic_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.critic_embed.parameters(), self.config.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), self.config.clip_grad_norm
            )

        self.critic_embed_optim.step()
        self.critic_optim.step()

        return metrics

