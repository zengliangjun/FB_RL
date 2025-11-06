from typing import List, Dict
import dataclasses

from envs.humenvs import configs as envs_configs
from base_envs import evals

@dataclasses.dataclass
class RewardEvaluationConfig(evals.BaseEvalConfig):
    _target_: str = "trainer.humenv.evals:RewardEvaluation"

    reward_tasks: List[str] = dataclasses.field(default_factory=list)
    env_kwargs: dict[str, str] = dataclasses.field(default_factory=dict)
    num_envs: int = 5
    num_episodes: int = 10
    num_inference_samples: int = 50_000

    def __post_init__(self):
        self.reward_tasks = [
                "move-ego-0-0",
                "jump-2",
                "move-ego-0-2",
                "move-ego-90-2",
                "move-ego-180-2",
                "rotate-x-5-0.8",
                "rotate-y-5-0.8",
                "rotate-z-5-0.8"
            ],
        self.env_kwargs = {"state_init": "Fall", "context": "spawn"},
        self.num_envs = 5,
        self.num_episodes = 10,
        self.num_inference_samples = 50_000


@dataclasses.dataclass
class TrackingEvaluationConfig(evals.BaseEvalConfig):
    _target_: str = "trainer.humenv.evals:TrackingEvaluation"

    motions: str = envs_configs.motions
    motions_root: str = envs_configs.motions_root
    env_kwargs: dict[str, str] = dataclasses.field(default_factory=dict)
    num_envs: int = 60

    def __post_init__(self):
        self.env_kwargs = {"state_init": "Default"}
        self.num_envs = 5 #60
