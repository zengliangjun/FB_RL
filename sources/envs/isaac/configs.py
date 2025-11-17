import os
import dataclasses
from typing import Optional, Any

cwd = os.getcwd()
motions_root = f"{cwd}/motions"
replay_root = "/workspace/data2/VSCODE/RL_MOTION_TRACKING/ISAACSIM45LAB2/whole_body_tracking/wandb-registry"


from base_envs import envs

@dataclasses.dataclass
class IsaacConfig(envs.BaseEnvConfig):
    _target_: str = "envs.isaac.env:IsaacWarp"

    workdir: Optional[str] = None
    args_cli: Optional[Any] = None
