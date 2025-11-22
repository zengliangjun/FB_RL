import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

from base import networks
from cpr.networks import utils
from cpr.networks.diffusion import configs, mlp

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.output_size // 2
        f = math.log(10000) / (half_dim - 1)
        f = torch.exp(torch.arange(half_dim, device=device) * -f)
        f = x * f[None, :]
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)

    def forward(self, x: torch.Tensor):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f

TIMEEMBED = {"fixed": SinusoidalPosEmb, "learned": LearnedPosEmb}


class IDQLDiffusion(nn.Module):
    """
    Diffusion model implementation for IDQL (Implicit Diffusion Q-Learning).

    Reference:
        IDQL: Implicit Diffusion Q-Learning - arXiv:2304.10573
    """
    def __init__(self, cfg: configs.IDQLConfig):
        super(IDQLDiffusion, self).__init__()
        self.config = cfg
        input_dimension = cfg.input_dimension
        condition_dimension = cfg.condition_dimension

        z_dimension = cfg.z_dimension
        out_dimension = cfg.out_dimension

        time_dimension = cfg.time_dimension
        time_embeding = cfg.time_embeding
        action_fn = cfg.action_fn

        hidden_dimension = cfg.hidden_dimension
        hidden_layers = cfg.hidden_layers


        # time embedding
        if time_embeding not in TIMEEMBED.keys():
            raise ValueError(
                f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}"
            )

        self.time_process = TIMEEMBED[time_embeding](1, time_dimension)
        self.time_encoder = mlp.MLP(time_dimension, [time_dimension * 2], time_dimension, ac_fn='mish')


        # decoder
        input_dim = input_dimension + time_dimension + condition_dimension

        self.decoder = mlp.MLPResNet(
            input_dim=input_dim,
            num_blocks=hidden_layers,
            hidden_dim=hidden_dimension,
            output_size=out_dimension,
            ac_fn=action_fn,
            use_layernorm=True,
            dropout_rate=0.1,
            condition_dim=z_dimension
        )

        print(self)

    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        """
        # Process time embedding
        if x_t.dim() == 3:
            time_embedding = self.time_process(time)
        else:
            time_embedding = self.time_process(time.view(-1, 1))

        time_embedding = self.time_encoder(time_embedding)

        # Concatenate conditioning if provided
        if condition is not None:  # self.config.condition_dimension
            x_t = torch.cat([x_t, condition], dim=-1)

        # Prepare input for decoder
        decoder_input = torch.cat([time_embedding, x_t], dim=-1)

        noise_pred = self.decoder(decoder_input, z)
        return noise_pred



def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape

    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps

    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps

    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
    """Discret VP noise schedule
    """
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1

    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

SCHEDULE = {
    'linear': linear_beta_schedule,
    'cosine': cosine_beta_schedule,
    'sigmoid': sigmoid_beta_schedule,
    'vp': vp_beta_schedule
}

class DiffusionActor(nn.Module):
    def __init__(self, cfg: configs.ActorDiffusionConfig):
        super(DiffusionActor, self).__init__()
        self.config = cfg

        # 获取维度参数
        state_dimension = cfg.state_dimension  # 状态维度
        action_dimension = cfg.action_dimension  # 动作维度
        z_dimension = cfg.z_dimension  # 技能向量维度
        out_dims: int = cfg.out_dimension # action_dimension

        hidden_dimension = cfg.hidden_dimension  # 隐藏层维度

        embedding_layers: int = cfg.embedding_layers  # 嵌入层数量

        if cfg.embedding_simple:
            self.embed_s = utils.simple_embedding(state_dimension, hidden_dimension, embedding_layers)
        else:
            self.embed_s = utils.residual_embedding(state_dimension, hidden_dimension, embedding_layers)

        self.policy = networks.NetProxy(cfg.diffusion_block)

        schedule = cfg.diffusion_schedule
        if schedule not in SCHEDULE.keys():
            raise ValueError(
                f"Invalid schedule '{schedule}'. Expected one of: {list(SCHEDULE.keys())}"
            )
        schedule = SCHEDULE[schedule]

        self.betas = schedule(cfg.diffusion_timesteps).to(cfg.device)
        self.alphas = (1 - self.betas).to(cfg.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(cfg.device)


    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.policy(xt, t, cond, z)

    def _predict_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        predict the noise
        """
        return self.policy(xt, t, cond, z)

    def _q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        sample noisy xt from x0, q(xt|x0), forward process
        """
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) \
            + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return xt

    def _p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        clip_sample: bool = False,
        ddpm_temperature: float = 1.
    ) -> torch.Tensor:
        """
        sample xt-1 from xt, p(xt-1|xt)
        """
        noise_pred = self.forward(xt, t, cond, z)

        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))

        xtm1 = alpha1 * (xt - alpha2 * noise_pred)

        noise = torch.randn_like(xtm1, device=xt.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)

        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1

    def _ddpm_sampler(
        self,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        num: int = 1
    ) -> torch.Tensor:
        """
        sample x0 from xT, reverse process
        """

        cond = self.embed_s(cond)
        _b = cond.shape[0]

        if _b > 1:

            out_dimension = self.config.diffusion_block.out_dimension

            x = torch.randn((_b, num, out_dimension), device=cond.device)
            cond = cond.unsqueeze(1).repeat_interleave(num, dim=1)
            z = z.unsqueeze(1).repeat_interleave(num, dim=1)

            for t in reversed(range(self.config.diffusion_timesteps)):
                x = self._p_sample(
                    xt=x,
                    t=torch.full((_b, num, 1), t, device=cond.device),
                    cond=cond,
                    z=z
                )
        else:
            out_dimension = self.config.diffusion_block.out_dimension
            x = torch.randn((num, out_dimension), device=cond.device)

            cond = cond.repeat(num, 1)
            z = z.repeat(num, 1)

            for t in reversed(range(self.config.diffusion_timesteps)):
                x = self._p_sample(
                    xt=x,
                    t=torch.full((num, 1), t, device=cond.device),
                    cond=cond,
                    z=z
                )
        return x

    def policy_loss(
        self,
        action: torch.Tensor,
        state: torch.Tensor,
        z_policy: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        '''
        calculate ddpm loss
        '''
        batch_size = action.shape[0]

        noise = torch.randn_like(action, device=action.device)
        t = torch.randint(0, self.config.diffusion_timesteps, (batch_size, ), device=action.device)

        xt = self._q_sample(action, t, noise)

        state = self.embed_s(state)

        noise_pred = self._predict_noise(xt, t, state, z_policy)
        if weight is None:
            return (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
        else:
            return (((noise_pred - noise) ** 2).sum(axis = -1) * weight).mean()

    def get_action(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        num: int = 1
    ) -> torch.Tensor:

        return self._ddpm_sampler(
            cond=state,
            z=z,
            num = num
        )


