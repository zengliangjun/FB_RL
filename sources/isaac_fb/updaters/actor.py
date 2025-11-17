import torch

from cpr.updaters.cpr import actorupdater

from base import models
from isaac_fb.updaters import configs
from isaac_fb.models import isaac

from fbutils.actor_post import ActorValueType

class ActorUpdater(actorupdater.ActorUpdater):

    config: configs.ActorConfig

    calcute: isaac.IsaacCalcute
    target_calcute: isaac.IsaacCalcute

    def __init__(self, cfg: configs.ActorConfig, model: models.BaseModel):
        super(ActorUpdater, self).__init__(cfg, model)

    def _calcute_reward_q(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        return self.calcute.reward_calcute(state=state, action =action, z_policy=z_policy)  # num_parallel x batch x (1 or n_bins)

    def _calcute_loss(self, inputs: dict, step: int) -> dict:
        state: torch.Tensor = inputs['state']  # 当前状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        ## 采样动作
        # 通过策略网络生成动作分布
        dist = self.calcute.act(state, z_policy, type = ActorValueType.DISTRIBUTION)  # TODO: 检查step参数的使用
        # 从分布中采样动作
        action = self.calcute.act_sample(dist)

        #
        Q_disc = self._calcute_disc(state, action, z_policy)
        Q_reward = self._calcute_reward_q(state, action, z_policy)
        Q_fb = self._calcute_q(state, action, z_policy)

        weight = Q_fb.abs().mean().detach() if self.config.scale_reg else 1.0
        Q_disc_loss = -Q_disc.mean() * self.config.reg_coeff * weight
        Q_reward_loss = -Q_reward.mean() * self.config.reword_coeff
        Q_loss = - Q_fb

        # 如果支持对数概率且配置了熵正则化
        # if hasattr(dist, 'log_prob'):  # and self.config.if_log_prob:
        if self.model.config.net_actor.boltzmann:
            # 计算动作的对数概率
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            # 应用熵正则化：Q值减去温度参数乘以对数概率
            prob_loss = self.config.temp * log_prob
            Q_loss += prob_loss

        # 策略损失：负的Q值均值（最大化Q值）
        actor_loss = Q_loss.mean() + Q_disc_loss + Q_reward_loss

        with torch.no_grad():
            output_metrics = {
                "actor/Q_dis": Q_disc.mean().detach(),
                "actor/Q_reward": Q_reward.mean().detach(),
                "actor/Q_fb": Q_fb.mean().detach(),
                "actor/loss": actor_loss.detach()
            }
            if self.model.config.net_actor.boltzmann:
                output_metrics["actor/log_prob"] = log_prob.mean().detach()

        return actor_loss, output_metrics  # 返回损失和空指标字典