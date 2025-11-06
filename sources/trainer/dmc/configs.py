from base_envs import trainers
from envs.dmc import configs as envs_configs
from cpr.updaters import configs as updaters_configs
from cpr.agents import configs as agents_configs
from cpr.buffers import configs as buffers_configs
from cpr.models import configs as models_configs

from cpr.networks import configs as net_configs

from typing import List

import dataclasses

#### Net
@dataclasses.dataclass
class  ForwardNetConfig(net_configs.SimpleForwardMapConfig):
    num_parallel = 2

    hidden_dimension = 1024
    embedding_layers = 2
    hidden_layers = 1

    optim_lr = 1e-4
    optim_weight_decay = 0

    target_tau = 0.01


@dataclasses.dataclass
class  BackwardNetConfig(net_configs.SimpleBackwardMapConfig):

    hidden_dimension = 256
    hidden_layers = 2

    optim_lr = 1e-4
    optim_weight_decay = 0

    target_tau = 0.01


@dataclasses.dataclass
class  ActorNetConfig(net_configs.SimpleActorConfig):

    hidden_dimension = 1024
    embedding_layers = 2
    hidden_layers = 1

    optim_lr = 1e-4
    optim_weight_decay = 0

    boltzmann: bool = False  # 是否使用Boltzmann策略
    # Boltzmann策略相关配置
    log_std_bounds: tuple = (-5.0, 2.0)  # 对数标准差边界，用于约束探索程度
    # 非Boltzmann策略配置
    #stddev_schedule: str = "0.2"  # 标准差调度策略，如 "linear(1,0.2,200000)"
    std: float = 0.2  # 标准差
    stddev_clip: float = 0.3  # 标准差裁剪值，控制探索噪声范围


#### Model
@dataclasses.dataclass
class  FBModelConfig(models_configs.FBModelConfig):

    state_dimension: int = 24
    action_dimension: int = 6
    z_dimension: int = 100

    optim_lr: float = 1e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01

    #_network_names_: List[str] = ["net_forward_map", "net_backward_map", "net_actor"]
    #_target_network_names_: List[str] = ["net_forward_map", "net_backward_map"]

    #_calcute_class_name_: str = "cpr.models.fb:FBCalcute"  # 目标网络字段名

    net_forward_map: ForwardNetConfig = ForwardNetConfig()
    net_backward_map: BackwardNetConfig = BackwardNetConfig()
    net_actor: ActorNetConfig = ActorNetConfig()


#### Updater
@dataclasses.dataclass
class  FBUpdaterConfig(updaters_configs.FBConfig):
    if_calcute_q: bool = False
    #q_loss_coef: float = 0

    pessimism_penalty: float = 0
    ortho_coef: float = 1
    clip_grad_norm: float = 0

@dataclasses.dataclass
class  ActorUpdaterConfig(updaters_configs.ActorConfig):
    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0


#### Agent
@dataclasses.dataclass
class FBAgentConfig(agents_configs.FBAgentConfig):
    name: str = "cprfb_agent"

    fb_updater: FBUpdaterConfig = FBUpdaterConfig()
    actor_updater: ActorUpdaterConfig = ActorUpdaterConfig()

    model: FBModelConfig = FBModelConfig()

    discount: float = 0.98
    batch_size: int = 1024

    def __post_init__(self):
        super().__post_init__()
        self._updater_names_ =  ["fb_updater", "actor_updater"]


#### buffer
@dataclasses.dataclass
class  DMCBufferConfig(buffers_configs.DMCBufferConfig):
    dataset_root: str = ""
    dataset_expl_agent: str = "rnd"
    domain_name: str = "walker"
    load_n_episodes : int = -1 ## all files

### Env
@dataclasses.dataclass
class  DMCConfig(envs_configs.DMCConfig):
    domain_name: str= "walker"
    task_name: str = "walk"
    max_episode_steps: int = envs_configs.DEFAULT_MAX_EPISODE_STEPS


@dataclasses.dataclass
class TrainerConfig(trainers.BaseTrainerConfig):
    _target_: str = "trainer.dmc.trainer:FBTrainer"  # 目标类路径

    name: str = "fb_trainer"
    seed: int = 40

    max_steps: int = 3000000
    eval_every_steps: int = 100000

    log_every_steps: int = 10000
    checkpoint_every_steps: int = 1000000


    env_config: DMCConfig = DMCConfig()
    agent_config: FBAgentConfig = FBAgentConfig()

    replay_buffer: DMCBufferConfig = DMCBufferConfig()

    def __post_init__(self):
        super().__post_init__()
        self._bufer_names_ =  ["replay_buffer"]
        self._eval_names_ = []
