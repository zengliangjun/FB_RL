from base_envs import envs
import dataclasses
import os


os.environ["OMP_NUM_THREADS"] = "1"


humenv = "/workspace/data2/VSCODE/MOTION/MATA/humenv"
motions = f"{humenv}/data_preparation/test_train_split/large1_small1_train_0.1.txt"
# motions = f"{humenv}/data_preparation/test_train_split/0-ACCAD_train_0.1.txt"
motions_root = f"{humenv}/data_preparation/humenv_amass"


@dataclasses.dataclass
class HumenvConfig(envs.BaseEnvConfig):
    _target_: str = "envs.humenvs.env:HumenvWarp"

    name: str = "humenv"

    num_envs: int = 50 # online_parallel_envs
    envinit_motions: str = motions
    envinit_motions_root: str = motions_root
