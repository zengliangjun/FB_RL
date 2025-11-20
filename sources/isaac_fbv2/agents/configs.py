
from base import agents
import dataclasses

@dataclasses.dataclass
class AgentConfig(agents.BaseAgentConfig):
    _target_: str = "isaac_fbv2.agents.encode:Agent"  # 目标类路径

    name: str = "encode_agent"

    discount: float = 0.98
    batch_size: int = 1024

    sample_goal_ratio: float =0.2
    expert_sample_ratio: float = 0.6

    relabel_ratio: float = 0.8

    seq_length: int = 8

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  [
                                 "vae_updater",
                                 "fb_updater",
                                 "actor_updater",
                                 "critic_updater",
                                 "reward_critic_updater",
                                 "discriminator_updater"]
