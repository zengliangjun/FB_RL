
from cpr.agents import configs
import dataclasses

@dataclasses.dataclass
class IsaacAgentConfig(configs.FBCPrAgentConfig):
    _target_: str = "isaac_fb.agents.isaac:Agent"  # 目标类路径

    name: str = "isaac_agent"
    discount: float = 0.98
    batch_size: int = 1024

    sample_goal_ratio: float =0.2
    expert_sample_ratio: float = 0.6

    relabel_ratio: float = 0.8

    seq_length: int = 8

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  ["fb_updater", "actor_updater", "critic_updater", "reward_critic_updater", "discriminator_updater"]
