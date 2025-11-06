from base_envs import envs
import dataclasses

ALL_TASKS = {
    "walker": [
        "walk",
        "run",
        "stand",
    ],
    "cheetah": ["walk", "run"],
    "quadruped": ["walk", "run"],
}


DEFAULT_MAX_EPISODE_STEPS = 300


@dataclasses.dataclass
class DMCConfig(envs.BaseEnvConfig):
    _target_: str = "envs.dmc.env:DMCWarp"

    domain_name: str= "walker"
    task_name: str = "walk"
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS

    def __post_init__(self):
        super().__post_init__()

        if self.task_name is None:
            if self.domain_name == "walker":
                self.task_name = "walk"
            elif self.domain_name == "cheetah":
                self.task_name = "run"
            elif self.domain_name == "pointmass":
                self.task_name = "reach_top_left"
            elif self.domain_name == "quadruped":
                self.task_name = "run"
            else:
                raise RuntimeError("Unsupported domain, you need to specify task_name")

        self.name = f"{self.domain_name}_{self.task_name}"

