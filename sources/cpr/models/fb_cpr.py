from torch import nn
import torch

from cpr.models import fb, configs

class FBCprCalcute(fb.FBCalcute):

    def __init__(self, cfg: configs.FBCprModelConfig):
        super(FBCprCalcute, self).__init__(cfg)

    def next_critic_calcute(self, inputs: dict, action: torch.Tensor):
        state = inputs["state"]
        next_state = inputs["next_state"]
        z_policy = inputs["z_policy"]
        return self.critic_calcute(next_state, action, z_policy)

    def critic_calcute(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        return self.critic(state, action, z_policy)


class  FBCprModel(fb.FBModel):

    def __init__(self, cfg: configs.FBCprModelConfig):
        super(FBCprModel, self).__init__(cfg)
