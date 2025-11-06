from torch import nn
import torch

from cpr.models import fb, configs

class FBCprCalcute(fb.FBCalcute):

    def __init__(self, cfg: configs.FBCprModelConfig):
        super(FBCprCalcute, self).__init__(cfg)


class  FBCprModel(fb.FBModel):

    def __init__(self, cfg: configs.FBCprModelConfig):
        super(FBCprModel, self).__init__(cfg)
