from base_envs import trainers
from base_envs.simple_buffer import configs as simple_configs

from envs.isaac import configs as envs_configs
from cpr.buffers import configs as buffers_configs
from isaac_fb.buffers import configs as isaac_buffers_configs

from cpr.networks import configs as net_configs
from isaac_fbv2.networks.residual import configs as netv2_configs
from isaac_fbv2.models import configs as models_configs
from isaac_fbv2.updaters import configs as updaters_configs
from isaac_fbv2.agents import configs as agents_configs

# from trainer.humenv import eval_configs

from typing import List

import dataclasses

#### Net
@dataclasses.dataclass
class  CriticEmbedConfig(netv2_configs.CriticEmbedConfig):
    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 2

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01


@dataclasses.dataclass
class  ForwardNetConfig(netv2_configs.ForwardMapBlockConfig):
    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 4

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01

@dataclasses.dataclass
class  CriticConfig(netv2_configs.CriticBlockConfig):
    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 4

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.005

@dataclasses.dataclass
class  RewardCriticConfig(netv2_configs.CriticBlockConfig):
    num_parallel: int = 2

    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 4

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.005



@dataclasses.dataclass
class  BackwardNetConfig(net_configs.SimpleBackwardMapConfig):
    hidden_dimension: int = 256
    hidden_layers: int = 1

    optim_lr: float = 1e-05
    optim_weight_decay: float = 0

    target_tau: float = 0.01
    norm_z: bool = True


@dataclasses.dataclass
class StateVAEConfig(netv2_configs.StateVAEConfig):
    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 4

    latent_dimension: int = 256

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0


@dataclasses.dataclass
class  ActorNetConfig(netv2_configs.ActorConfig):
    hidden_dimension: int = 1024
    embedding_layers: int = 4
    hidden_layers: int = 6

    latent_dimension: int = 256

    optim_lr: float = 3e-4
    optim_weight_decay: float = 0

    boltzmann: bool = False  # 是否使用Boltzmann策略
    # Boltzmann策略相关配置
    log_std_bounds: tuple = (-5.0, 2.0)  # 对数标准差边界，用于约束探索程度
    # 非Boltzmann策略配置
    #stddev_schedule: str = "0.2"  # 标准差调度策略，如 "linear(1,0.2,200000)"
    std: float = 0.2  # 标准差
    stddev_clip: float = 0.3  # 标准差裁剪值，控制探索噪声范围


@dataclasses.dataclass
class  DiscriminatorConfig(net_configs.SimpleDiscriminatorConfig):
    hidden_dimension: int = 1024
    hidden_layers: int = 2

    optim_lr: float = 1e-5
    optim_weight_decay: float = 0


#### Model
@dataclasses.dataclass
class  FBModelConfig(models_configs.ModelConfig):

    state_dimension: int = 64
    action_dimension: int = 29
    z_dimension: int = 256

    privileges_dimension: int = 607
    history_horizon: int = 4

    optim_lr: float = 1e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01

    #_network_names_: List[str] = ["net_forward_map", "net_backward_map", "net_actor"]
    #_target_network_names_: List[str] = ["net_forward_map", "net_backward_map"]

    #_calcute_class_name_: str = "cpr.models.fb:FBCalcute"  # 目标网络字段名
    net_critic_embed: CriticEmbedConfig = CriticEmbedConfig()

    net_forward_map: ForwardNetConfig = ForwardNetConfig()
    net_critic: CriticConfig = CriticConfig()
    net_reward_critic: RewardCriticConfig = RewardCriticConfig()

    net_actor_vae: StateVAEConfig = StateVAEConfig()
    net_actor: ActorNetConfig = ActorNetConfig()

    net_discriminator: DiscriminatorConfig = DiscriminatorConfig()
    net_backward_map: BackwardNetConfig = BackwardNetConfig()



#### Updater
@dataclasses.dataclass
class  FBUpdaterConfig(updaters_configs.FBConfig):
    if_calcute_q: bool = True
    q_loss_coef: float = 0.1
    ortho_coef: float = 100

    pessimism_penalty: float = 0
    clip_grad_norm: float = 0

@dataclasses.dataclass
class  ActorUpdaterConfig(updaters_configs.ActorConfig):

    scale_reg: bool = False
    reg_coeff: float = 0.05
    reword_coeff: float = 0.02

    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0

@dataclasses.dataclass
class  CriticUpdaterConfig(updaters_configs.CriticConfig):
    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0

@dataclasses.dataclass
class  RewardUpdaterConfig(updaters_configs.RewardCriticConfig):
    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0


@dataclasses.dataclass
class  DiscriminatorUpdaterConfig(updaters_configs.DiscriminatorConfig):
    grad_penalty: float = 10.0

    pessimism_penalty: float = 0.5
    clip_grad_norm: float = 0


@dataclasses.dataclass
class  VAEUpdaterConfig(updaters_configs.VAEConfig):
    kl_loss_coef: float = 1
    clip_grad_norm: float = 0

#### Agent
@dataclasses.dataclass
class FBAgentConfig(agents_configs.AgentConfig):
    vae_updater: VAEUpdaterConfig = VAEUpdaterConfig()
    fb_updater: FBUpdaterConfig = FBUpdaterConfig()
    actor_updater: ActorUpdaterConfig = ActorUpdaterConfig()
    critic_updater: CriticUpdaterConfig = CriticUpdaterConfig()
    reward_critic_updater: RewardUpdaterConfig = RewardUpdaterConfig()
    discriminator_updater: DiscriminatorUpdaterConfig = DiscriminatorUpdaterConfig()

    model: FBModelConfig = FBModelConfig()

    discount: float = 0.98
    batch_size: int = 1024 // 8

    sample_goal_ratio: float = 0.2
    expert_sample_ratio: float = 0.6

    relabel_ratio: float = 0.8

    seq_length: int = 8


#### buffer
@dataclasses.dataclass
class  ZBufferConfig(simple_configs.BufferConfig):
    capacity: int = 10000
    dim: int = 256
    # device = "cuda"

@dataclasses.dataclass
class  MotionBufferConfig(isaac_buffers_configs.MotionBufferConfig):

    motions_root: str = envs_configs.motions_root
    # device: str = "cuda"
    seq_length: int = 8
    history_horizon: int = 4

    prioritization: bool = True
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2

@dataclasses.dataclass
class  RolloutBufferConfig(buffers_configs.DictBufferConfig):
    capacity: int = 5_120_000 // 8  # b * 10 * episode len
    # device: str = "cuda"


### Env
@dataclasses.dataclass
class  HumEnvConfig(envs_configs.IsaacConfig):
    pass


@dataclasses.dataclass
class TrainerConfig(trainers.BaseTrainerConfig):
    _target_: str = "isaac_fb.train.isaac:IsaacTrainer"  # 目标类路径

    name: str = "isaac_fbv2_trainer"
    seed: int = 0

    max_steps: int = 192_000 * 8
    seed_steps: int = 100

    log_every_steps: int = 1 * 8 #2_000
    checkpoint_every_steps: int = 2_000 * 8
    eval_every_steps: int = 2_000 * 8

    one_step_collect_iters: int = 1
    one_step_update_iters: int = 16

    ##
    use_mix_rollout: bool = True
    update_z_every_env_steps: int = 150

    env_config: HumEnvConfig = HumEnvConfig()
    agent_config: FBAgentConfig = FBAgentConfig()

    expert_buffer: MotionBufferConfig = MotionBufferConfig()
    rollout_buffer: RolloutBufferConfig = RolloutBufferConfig()
    z_buffer: ZBufferConfig = ZBufferConfig()

    # eval_reward: eval_configs.RewardEvaluationConfig = eval_configs.RewardEvaluationConfig()
    # eval_tracking: eval_configs.TrackingEvaluationConfig = eval_configs.TrackingEvaluationConfig()

    def __post_init__(self):
        super().__post_init__()
        self._bufer_names_ =  ["expert_buffer", "rollout_buffer", "z_buffer"]
        self._eval_names_ = []


