import os
import sys
import os.path as osp
import json
import dataclasses

script_dir = osp.abspath(osp.dirname(__file__))
workroot_dir = osp.abspath(osp.join(script_dir, "../.."))
os.chdir(workroot_dir)

source_dir = f"{workroot_dir}/sources"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from base_envs import trainers
from trainer.humenv import configs, meta_configs

from cpr.agents import configs as agents_configs
from cpr.models import configs as models_configs
from cpr.updaters import configs as cpr_configs


@dataclasses.dataclass
class  ActorUpdaterConfig(cpr_configs.ActorConfig):

    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0


@dataclasses.dataclass
class FBAgentConfig(agents_configs.FBAgentConfig):
    name: str = "fb_agent_meta"

    fb_updater: configs.FBUpdaterConfig = configs.FBUpdaterConfig()
    actor_updater: ActorUpdaterConfig = ActorUpdaterConfig()

    model: meta_configs.FBModelConfig = meta_configs.FBModelConfig()

    discount: float = 0.98
    batch_size: int = 1024

    sample_goal_ratio: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  ["fb_updater", "actor_updater"]

if __name__ == "__main__":

    config = configs.TrainerConfig()
    #  config.env_config.num_envs = 2
    config.agent_config = FBAgentConfig()
    config.name = config.name.replace("cpr", "meta")

    config._eval_names_ = [] # ["eval_tracking"]

    trainer = trainers.TrainerProxy(config)
    trainer.train()

