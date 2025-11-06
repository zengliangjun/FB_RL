import dataclasses
import torch
import numpy as np
from typing import Tuple, Dict, Union

from base import models, configs


@dataclasses.dataclass
class BaseEnvConfig(configs.BaseConfig):
    name: str = ""


class EnvProxy():

    def __new__(self, cfg: BaseEnvConfig):
        return cfg.instantiate_from_config()


class Env():

    def __init__(self, cfg: BaseEnvConfig):
        self.config = cfg

    def reset(self) -> Tuple[torch.Tensor, Dict[str, object]]:
        pass

    def timestep(self) -> torch.Tensor:
        pass

    def sample_action(self) -> torch.Tensor:
        pass

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> \
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor, Dict[str, object]]:

        pass

    def end(self, *args, **kwargs):
        pass
