from base import agents


import dataclasses


@dataclasses.dataclass
class FBAgentConfig(agents.BaseAgentConfig):
    _target_: str = "cpr.agents.fb_agent:FBAgent"  # 目标类路径

    name: str = "fb_agent"

    discount: float = 0.98
    batch_size: int = 1024
    sample_goal_ratio: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  ["fb_updater", "actor_updater"]



@dataclasses.dataclass
class FBCPrAgentConfig(agents.BaseAgentConfig):
    _target_: str = "cpr.agents.fb_cpr_agents:FBAgent"  # 目标类路径

    name: str = "cpr_fbcpr_agent"
    discount: float = 0.98
    batch_size: int = 1024

    sample_goal_ratio: float =0.2
    expert_sample_ratio: float = 0.6

    relabel_ratio: float = 0.8

    seq_length: int = 8

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  ["fb_updater", "actor_updater", "critic_updater", "discriminator_updater"]
