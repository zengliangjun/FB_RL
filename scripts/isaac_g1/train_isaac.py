import isaac_init

import os
import sys
import os.path as osp

script_dir = osp.abspath(osp.dirname(__file__))
workroot_dir = osp.abspath(osp.join(script_dir, "../.."))
os.chdir(workroot_dir)


source_dir = f"{workroot_dir}/sources"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)


import envs.isaac
from base_envs import trainers
from isaac_fb.train import configs


if __name__ == "__main__":

    cfg: configs.TrainerConfig  = configs.TrainerConfig()
    cfg.env_config.args_cli = isaac_init.args_cli
    cfg.env_config.args_cli.task = "FB-G129dof-v0"

    trainer = trainers.TrainerProxy(cfg)
    trainer.train()

