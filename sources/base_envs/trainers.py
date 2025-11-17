
import dataclasses
import os
import os.path as osp
from datetime import datetime
import random
import torch
import numpy as np
from loguru import logger as ulogger
from typing import List
import time

from base import agents, configs
from base_envs import envs, buffers, evals
from fbutils import logger


@dataclasses.dataclass
class BaseTrainerConfig(configs.BaseConfig):
    name: str = "base_trainer"
    seed: int = 0

    max_steps: int = 3000000
    eval_every_steps: int = 100000

    log_every_steps: int = 10000
    checkpoint_every_steps: int = 1000000

    env_config: envs.BaseEnvConfig = None
    agent_config: agents.BaseAgentConfig = None

    _bufer_names_: List[str] = dataclasses.field(default_factory=list)   # 网络字段名
    _eval_names_: List[str] = dataclasses.field(default_factory=list)   # 网络字段名

    def __post_init__(self):
        super().__post_init__()
        self.name = f"{self.env_config.name}_{self.agent_config.name}"

    def bufer_names(self):
        return self._bufer_names_

    def bufer_config(self, name):
        cfg = getattr(self, name)
        assert cfg is not None
        assert hasattr(cfg, "_target_")
        assert isinstance(cfg, buffers.BaseBufferConfig)
        return cfg

    def eval_names(self):
        return self._eval_names_

    def eval_config(self, name):
        cfg = getattr(self, name)
        assert cfg is not None
        assert hasattr(cfg, "_target_")
        assert isinstance(cfg, evals.BaseEvalConfig)
        return cfg


class TrainerProxy():

    def __new__(self, cfg: BaseTrainerConfig):
        return cfg.instantiate_from_config()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class BaseTrainer:

    workdir: str
    env: envs.Env
    agent: agents.BaseAgent

    def __init__(self, cfg: BaseTrainerConfig):
        self.config = cfg

        self._preinit_workdir()
        set_seed_everywhere(cfg.seed)

        ##
        self.env  = envs.EnvProxy(cfg.env_config)
        ulogger.info(f"init_env.")
        self.agent = agents.AgentProxy(cfg.agent_config)
        ulogger.info(f"init_agent.")

        ## bufer
        for name in cfg.bufer_names():
            bufer_cfg = cfg.bufer_config(name)
            buffer = buffers.BufferProxy(bufer_cfg)
            setattr(self, name, buffer)
            ulogger.info(f"init_buffer: {name}")

        ## eval
        for name in cfg.eval_names():
            env_cfg = cfg.eval_config(name)
            eval = evals.EvalProxy(env_cfg)
            setattr(self, name, eval)
            ulogger.info(f"init eval: {name}")

        self._postinit_dump()

    def _preinit_workdir(self):
        date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        workdir = osp.join(os.getcwd(), "logs", self.config.name, f"run_{date}")
        os.makedirs(workdir)
        self.workdir = workdir
        setattr(self.config.env_config, "workdir", workdir)

        ulogger.info(f"init_workdir: {workdir}.")

        self.logger = logger.Logger(workdir,
                             use_tb=True,
                             use_wandb=False,
                             use_hiplog=False)

    def _postinit_dump(self):
        path = f"{self.workdir}/config.json"
        self.config.save(path)

    ##
    def _train_save(self, infos):
        step, = infos

        if step % self.config.checkpoint_every_steps != 0 and step != self.config.max_steps - 1:
            return

        self.agent.save(self.workdir, step)

    ##
    def _train_update_metrics(self, total_metrics, metrics):
        for k, v in metrics.items():
            if k in total_metrics:
                total_metrics[k] += v
            else:
                if isinstance(v, torch.Tensor):
                    total_metrics[k] = v.clone()
                else:
                    total_metrics[k] = v

    def _train_logs(self, infos):
        step, start_time, fps_start_time, num_updates, total_metrics = infos

        if step ==0:
            return False

        if step % self.config.log_every_steps != 0:
            return False

        if total_metrics is None:
            return False

        logger_pad = 35

        logger_dict = {}
        for k in sorted(list(total_metrics.keys())):
            tmp = total_metrics[k] / num_updates
            logger_dict[k] = tmp

        time_dict = {}
        time_dict["duration [minutes]"] = (time.time() - start_time) / 60
        time_dict["FPS"] = num_updates / (time.time() - fps_start_time)

        ulogger_buffer = f"\n{' step:':>{logger_pad}} {step}\n"
        for k in sorted(list(logger_dict.keys())):
            ulogger_buffer += f"{k:>{logger_pad}}: {logger_dict[k]}\n"

        for k in sorted(list(time_dict.keys())):
            ulogger_buffer += f"{k:>{logger_pad}}: {time_dict[k]}\n"

        ulogger.info(ulogger_buffer)

        logger_dict.update(time_dict)
        self.logger.log_metrics(logger_dict, step, ty='train')

        return True
