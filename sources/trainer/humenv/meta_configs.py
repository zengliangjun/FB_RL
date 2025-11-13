
import dataclasses


from cpr.models import models_meta as meta_configs
from trainer.humenv import configs as humenv_configs


@dataclasses.dataclass
class  FBModelConfig(meta_configs.MetaFBConfig):

    optim_lr: float = 1e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01

    net_forward_map: humenv_configs.ForwardNetConfig = humenv_configs.ForwardNetConfig()
    net_backward_map: humenv_configs.BackwardNetConfig = humenv_configs.BackwardNetConfig()
    net_actor: humenv_configs.ActorNetConfig = humenv_configs.ActorNetConfig()

    def __post_init__(self):
        super().__post_init__()

        self._network_names_  = ["net_forward_map", "net_backward_map", "net_actor"]
        self._target_network_names_ = ["net_forward_map", "net_backward_map"]

        ## metaconfig
        self.metaconfig.obs_dim = 358
        self.metaconfig.action_dim = 69
        self.metaconfig.norm_obs = True
        self.metaconfig.actor_std = 0.2
        self.metaconfig.seq_length = 8
        # archi
        # the config of the model trained in the paper
        model, hidden_dim, hidden_layers = "simple", 1024, 2
        self.metaconfig.archi.z_dim = 256
        self.metaconfig.archi.b.norm = 1
        self.metaconfig.archi.norm_z = 1
        self.metaconfig.archi.f.hidden_dim = hidden_dim
        self.metaconfig.archi.b.hidden_dim = 256
        self.metaconfig.archi.actor.hidden_dim = hidden_dim
        # self.metaconfig.archi.critic.hidden_dim = hidden_dim
        self.metaconfig.archi.f.hidden_layers = hidden_layers
        self.metaconfig.archi.b.hidden_layers = 1
        self.metaconfig.archi.actor.hidden_layers = hidden_layers
        # self.metaconfig.archi.critic.hidden_layers = hidden_layers
        self.metaconfig.archi.f.model = model
        self.metaconfig.archi.actor.model = model
        self.metaconfig.device = "cuda:0"



@dataclasses.dataclass
class  CPRModelConfig(meta_configs.MetaCPRConfig):

    optim_lr: float = 1e-4
    optim_weight_decay: float = 0

    target_tau: float = 0.01

    net_forward_map: humenv_configs.ForwardNetConfig = humenv_configs.ForwardNetConfig()
    net_backward_map: humenv_configs.BackwardNetConfig = humenv_configs.BackwardNetConfig()
    net_actor: humenv_configs.ActorNetConfig = humenv_configs.ActorNetConfig()

    net_critic: humenv_configs.CriticConfig = humenv_configs.CriticConfig()
    net_discriminator: humenv_configs.DiscriminatorConfig = humenv_configs.DiscriminatorConfig()

    def __post_init__(self):
        super().__post_init__()

        self._network_names_  = ["net_forward_map", "net_backward_map", "net_actor", "net_critic", "net_discriminator"]
        self._target_network_names_ = ["net_forward_map", "net_backward_map", "net_critic"]



