import os
import sys
import os.path as osp
import json

script_dir = osp.abspath(osp.dirname(__file__))
workroot_dir = osp.abspath(osp.join(script_dir, "../.."))
os.chdir(workroot_dir)

source_dir = f"{workroot_dir}/sources"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from base_envs import trainers
from trainer.humenv import configs


if __name__ == "__main__":
    config = configs.TrainerConfig()
    #  config.env_config.num_envs = 2
    config._eval_names_ = ["eval_tracking"]

    trainer = trainers.TrainerProxy(config)
    trainer.train()

