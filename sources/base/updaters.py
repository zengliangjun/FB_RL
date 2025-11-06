
import torch
from typing import Dict
import dataclasses


from base import models, configs


@dataclasses.dataclass
class BaseUpdaterConfig(configs.BaseConfig):
    pass


class UpdaterProxy():

    def __new__(self, cfg: BaseUpdaterConfig, model: models.BaseModel):
        return cfg.instantiate_from_config(model = model)

class Updater():
    def __init__(self, cfg: BaseUpdaterConfig, model: models.BaseModel):
        self.config = cfg
        self.model = model

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:
        pass

    def save_dict(self, collect_dict: dict, prefix: str):
        pass

    def resume_dict(self, collect_dict: dict, prefix: str):
        pass
