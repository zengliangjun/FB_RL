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
    _target_: str = "isaac_fb.buffers.isaac_buffer:MotionBuffer"

    device: str = "cpu"

    motions_root: Optional[str] = None
    robot_name: str = "g1"

    fps: int = 50
    seq_length: int = 8
    history_horizon: int = 4

    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2