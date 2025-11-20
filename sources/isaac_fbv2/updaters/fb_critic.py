from cpr.updaters import fbupdater
from isaac_fbv2.models import encode
from isaac_fbv2.updaters import configs

from torch.optim import Optimizer
import torch
from typing import Dict

class FBUpdater(fbupdater.FBUpdater):

    critic_embed_optim: Optimizer

    def __init__(self, cfg: configs.FBConfig, model: encode.Model):
        super(FBUpdater, self).__init__(cfg, model)
        self.critic_embed_optim = getattr(model, "critic_embed_optim")


    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        # 优化FB网络
        self.critic_embed_optim.zero_grad(set_to_none=True)
        self.forward_map_optim.zero_grad(set_to_none=True)
        self.backward_map_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.critic_embed.parameters(), self.config.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.forward_map.parameters(), self.config.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.backward_map.parameters(), self.config.clip_grad_norm
            )

        self.critic_embed_optim.step()
        self.forward_map_optim.step()
        self.backward_map_optim.step()

        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_critic_embed_optim"] = self.critic_embed_optim.state_dict()
        collect_dict[f"{prefix}_forward_map_optim"] = self.forward_map_optim.state_dict()
        collect_dict[f"{prefix}_backward_map_optim"] = self.backward_map_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.critic_embed_optim.load_state_dict(collect_dict[f"{prefix}_critic_embed_optim"])
        self.forward_map_optim.load_state_dict(collect_dict[f"{prefix}_forward_map_optim"])
        self.backward_map_optim.load_state_dict(collect_dict[f"{prefix}_backward_map_optim"])

