import dataclasses
import torch
from base import configs

@dataclasses.dataclass
class BaseBufferConfig(configs.BaseConfig):
    name: str = ""

class BufferProxy():

    def __new__(self, cfg: BaseBufferConfig):
        return cfg.instantiate_from_config()

class BufferBase():

    def __init__(self, cfg: BaseBufferConfig):
        self.config = cfg

    def __len__(self) -> int:
        pass

    def empty(self) -> bool:
        pass

    def sample(self, *args, **kwargs) -> torch.Tensor:
        pass

    def load(self, *args, **kwargs):
        pass

class Rollout(BufferBase):

    def __init__(self, cfg: BaseBufferConfig):
        super(Rollout, self)

    def extend(self, *args, **kwargs) -> None:
        pass

    def save(self, *args, **kwargs):
        pass
