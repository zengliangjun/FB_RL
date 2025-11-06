import torch
import dataclasses
from typing import Union

from base_envs import buffers

@dataclasses.dataclass
class BufferConfig(buffers.BaseBufferConfig):
    """
    Dataclass for replay buffer configuration.

    Args:
        max_size: maximum size of the replay buffer
        batch_size: batch size for sampling from the replay buffer
    """
    _target_: str = "base_envs.simple_buffer.buffer:ZBuffer"

    capacity: int = 10000
    dim: int = 256
    device: str = "cpu"
