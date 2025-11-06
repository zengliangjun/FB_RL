
import dataclasses
from base_envs import buffers
from typing import Optional

@dataclasses.dataclass
class MotionBufferConfig(buffers.BaseBufferConfig):
    """
    Dataclass for replay buffer configuration.

    Args:
        max_size: maximum size of the replay buffer
        batch_size: batch size for sampling from the replay buffer
    """
    _target_: str = "cpr.buffers.buffer:MotionBuffer"

    device: str = "cpu"

    motions: Optional[str] = None
    motions_root: Optional[str] = None

    seq_length: int = 8

    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2



@dataclasses.dataclass
class DictBufferConfig(buffers.BaseBufferConfig):
    """
    Dataclass for replay buffer configuration.

    Args:
        max_size: maximum size of the replay buffer
        batch_size: batch size for sampling from the replay buffer
    """
    _target_: str = "cpr.buffers.buffer:DictBuffer"
    device: str = "cpu"

    capacity: int = 10000



@dataclasses.dataclass
class DMCBufferConfig(buffers.BaseBufferConfig):
    """
    Dataclass for replay buffer configuration.

    Args:
        max_size: maximum size of the replay buffer
        batch_size: batch size for sampling from the replay buffer
    """
    _target_: str = "cpr.buffers.buffer:DMCBuffer"
    device: str = "cpu"

    dataset_root: str = ""
    dataset_expl_agent: str = ""
    domain_name: str = ""
    load_n_episodes : int = -1 ## all files
