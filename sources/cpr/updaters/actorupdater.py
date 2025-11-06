from base import updaters, models

from cpr.updaters import configs
from cpr.models import fb
from utils.actor_post import ActorValueType

from typing import Dict, Union
import torch
from torch.nn import functional as F
from torch.optim import Optimizer


class ActorUpdater(updaters.Updater):

    config: configs.ActorConfig

    calcute: fb.FBCalcute
    target_calcute: fb.FBCalcute

    actor_optim: Optimizer


    def __init__(self, cfg: configs.ActorConfig, model: models.BaseModel):
        super(ActorUpdater, self).__init__(cfg, model)

        self.calcute = model.calcute
        self.target_calcute = model.target_calcute

        self.actor_optim = getattr(model, "actor_optim")

    def _calcute_q(self, state: torch.Tensor, action: torch.Tensor, z_policy: torch.Tensor):
        """计算Q值

        根据状态、动作和技能向量计算Q值。
        支持多种Q值计算方式：单网络、双网络、集成网络。

        Args:
            state: 状态张量，形状为 [batch_size, state_dimension]
            action: 动作张量，形状为 [batch_size, action_dimension]
            z_policy: 技能向量，形状为 [batch_size, z_dimension]

        Returns:
            Q值张量，形状为 [batch_size] 或 [batch_size, 1]
        """
        # 计算前向表示
        F = self.calcute.forward_representation(state, action, z_policy)

        # 处理双网络情况（如TD3）
        if F is [tuple, list] and len(F) == 2:
            # 分别计算两个网络的Q值，取最小值（悲观估计）
            Q = [torch.einsum('sd, sd -> s', Fi, z_policy) for Fi in F]
            Q = torch.min(Q[0], Q[1])  # 双Q学习，取最小值避免过度乐观
        else:
            # 处理集成网络情况
            if len(F.shape) == 3:
                # 集成网络：多个并行网络计算Q值
                Q = torch.einsum('psd, sd -> ps', F, z_policy)  # p x batch_size

                # 使用不确定性估计计算最终Q值
                assert hasattr(self.calcute, 'get_targets_uncertainty')
                # 参考：metamotivo/metamotivo/fb/agent.py : update_td3_actor
                Q = self.calcute.get_targets_uncertainty(
                    Q, pessimism_penalty=self.config.pessimism_penalty)
            else:
                # 单网络情况：直接计算Q值
                # Q = (F * z_policy).sum(-1)  # 等价计算方式
                Q = torch.einsum('sd, sd -> s', F, z_policy)  # batch_size

        return Q

    def _calcute_loss(self, inputs: dict, step: int) -> dict:
        """计算策略损失

        根据输入数据计算策略网络的损失。
        通过最大化Q值来优化策略。

        Args:
            inputs: 输入数据字典，包含状态、技能向量等
            step: 当前训练步数，用于策略调度

        Returns:
            actor_loss: 策略损失值
            metrics: 训练指标字典
        """
        state: torch.Tensor = inputs['state']  # 当前状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        ## 采样动作
        # 通过策略网络生成动作分布
        dist = self.calcute.act(state, z_policy, type = ActorValueType.DISTRIBUTION)  # TODO: 检查step参数的使用
        # 从分布中采样动作
        action = self.calcute.act_sample(dist)

        # 计算当前状态-动作对的Q值
        Q = self._calcute_q(state, action, z_policy)
        Q_loss = - Q

        # 如果支持对数概率且配置了熵正则化
        if self.model.config.net_actor.boltzmann:  # and self.config.if_log_prob:
            # 计算动作的对数概率
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            prob_loss = self.model.config.net_actor.temp * log_prob
            # 应用熵正则化：Q值减去温度参数乘以对数概率
            Q_loss += prob_loss

        # 策略损失：负的Q值均值（最大化Q值）
        actor_loss = Q_loss.mean()

        with torch.no_grad():
            output_metrics = {
                "actor/Q_fb": Q.mean().detach(),
                "actor/loss": actor_loss.detach()
            }

            if self.model.config.net_actor.boltzmann:
                output_metrics["actor/log_prob"] = log_prob.mean().detach()

        return actor_loss, output_metrics  # 返回损失和空指标字典


    def _calcute_loss_org(self, inputs: dict, step: int) -> dict:
        state: torch.Tensor = inputs['state']  # 当前状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        action = self.calcute.act(state, z_policy, type = ActorValueType.SAMPLE)  # TODO: 检查step参数的使用

        Fs = self.calcute.forward_representation(state, action, z_policy)

        Qs = (Fs * z_policy).sum(-1)  # num_parallel x batch
        Q = self.calcute.get_targets_uncertainty(Qs, self.config.pessimism_penalty)  # batch
        actor_loss = -Q.mean()


        return actor_loss, {"actor/loss": actor_loss.detach(), "actor/Q_fb": Q.mean().detach()}

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        # 优化FB网络
        self.actor_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:

            torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), self.config.clip_grad_norm
            )

        self.actor_optim.step()

        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_actor_optim"] = self.actor_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.actor_optim.load_state_dict(collect_dict[f"{prefix}_actor_optim"])
