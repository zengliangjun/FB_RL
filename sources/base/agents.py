
import torch.nn as nn
import torch
from typing import List, Optional

import dataclasses
import os.path as osp
import os
import json
from safetensors.torch import save_model as safetensors_save_model
from safetensors.torch import load_model as safetensors_load_model

from loguru import logger as ulogger

from base import models, configs, updaters

@dataclasses.dataclass
class BaseAgentConfig(configs.BaseConfig):
    name: str = "base_agent"
    _updater_names_: List[str] = dataclasses.field(default_factory=list)  # 目标类路径

    resume_path: Optional[str] = None

    def updater_names(self):
        return self._updater_names_

    def updater_config(self, name):
        cfg = getattr(self, name)
        assert cfg is not None
        assert hasattr(cfg, "_target_")
        assert isinstance(cfg, updaters.BaseUpdaterConfig)
        return cfg


class AgentProxy():

    def __new__(self, cfg: BaseAgentConfig):
        return cfg.instantiate_from_config()


    @staticmethod
    def resume_from_dir(model_folder: str, device: str | None = None):

        jsonfile = osp.join(model_folder, "config.json")
        model_file = osp.join(model_folder, "model.safetensors")

        assert osp.exists(jsonfile)
        assert osp.exists(model_file)


        target = configs.BaseConfig.load(jsonfile)

        configs.config_update(target, "resume_path", model_folder)
        if device is not None:
            configs.config_update(target, "device", device)

        agent = AgentProxy(target)
        return agent



class BaseAgent():

    model: models.BaseModel

    def __init__(self, cfg: BaseAgentConfig):
        super(BaseAgent, self).__init__()
        self.config = cfg

        assert hasattr(self.config, "model")

        try:
            self.model = models.ModelsProxy(getattr(self.config, "model"))
        except Exception as e:
            ulogger.error(f"Error in model creation: {e}")
            from cpr.models import models_meta
            self.model = models_meta.ModelsProxy(getattr(self.config, "model"))

        for name in cfg.updater_names():
            updater_cfg = cfg.updater_config(name)
            updater = updaters.UpdaterProxy(updater_cfg, self.model)
            setattr(self, name, updater)

        if hasattr(self.config, "resume_path") and self.config.resume_path is not None:
            if osp.exists(self.config.resume_path):
                self._resume(self.config.resume_path)

    def save(self, workdir, step):
        output_folder = f"{workdir}/checkpoints/model_{step}"
        os.makedirs(output_folder, exist_ok=True)

        ## config
        jsonfile = osp.join(output_folder, "config.json")
        if osp.exists(jsonfile):
            os.remove(jsonfile)

        self.config.save(jsonfile)

        ## updater
        optimizers = {}
        for name in self.config.updater_names():
            updater = getattr(self, name)
            updater.save_dict(optimizers, prefix = f"updater_{name}")

        torch.save(optimizers, osp.join(output_folder, "optimizers.pth"))

        ## model
        safetensors_save_model(self.model, osp.join(output_folder, "model.safetensors"))

    def resume(self, workdir, step):
        checkpoints_folder = f"{workdir}/checkpoints/model_{step}"
        assert osp.exists(checkpoints_folder)
        self._resume(checkpoints_folder)

    def _resume(self, checkpoints_folder):
        optimizer_file = osp.join(checkpoints_folder, "optimizers.pth")
        model_file = osp.join(checkpoints_folder, "model.safetensors")

        ## updater
        if osp.exists(optimizer_file):
            optimizers = torch.load(optimizer_file, weights_only=True)
            for name in self.config.updater_names():
                updater = getattr(self, name)
                updater.resume_dict(optimizers, prefix = f"updater_{name}")

            ulogger.info(f"resume optimizers: {optimizer_file}")

        ## model
        if osp.exists(model_file):
            safetensors_load_model(self.model, model_file, device=self.model.config.device)
            self.model.init_target()
            ulogger.info(f"resume model: {model_file}")
