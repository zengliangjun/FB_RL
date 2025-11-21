import torch

from typing import Dict, Union, Tuple

from base import models
from cpr.updaters import fbupdater
from cpr.updaters import configs

from fbutils.actor_post import ActorValueType

class FBUpdater(fbupdater.FBUpdater):

    config: configs.FBConservativeConfig

    def __init__(self, cfg: configs.FBConservativeConfig, model: models.BaseModel):
        super(FBUpdater, self).__init__(cfg, model)


        # total_action_samples must be divisible by 4
        assert (cfg.conservative_ood_action_weight % 0.25 == 0) & (
            0 < cfg.conservative_ood_action_weight <= 1
        )  # ood_action_weight must be divisible by 0.25
        self.total_action_samples = cfg.conservative_total_action_samples
        self.ood_action_samples = int(self.total_action_samples * cfg.conservative_ood_action_weight)
        self.actor_action_samples = int(
            (self.total_action_samples - self.ood_action_samples) / 3
        )
        assert (
            self.ood_action_samples + (3 * self.actor_action_samples)
            == self.total_action_samples
        )

        # lagrange multiplier
        self.critic_log_alpha = torch.zeros(1, requires_grad=True, device=self.model.config.device)

        # optimizer
        self.critic_alpha_optimizer = torch.optim.Adam(
            [self.critic_log_alpha], lr=cfg.conservative_critic_learning_rate
        )


    def _calcute_penalty_F(self, inputs: dict,
        items: dict, next_items: dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        state: torch.Tensor = inputs['state']  # 当前状态
        action: torch.Tensor = inputs['action']  # 当前动作
        next_state: torch.Tensor = inputs['next_state']  # 下一状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        batch_size = state.shape[0]

        with torch.no_grad():
            # repeat observations, next_observations, zs, and Bs
            # we fold the action sample dimension into the batch dimension
            # to allow the tensors to be passed through F and B; we then
            # reshape the output back to maintain the action sample dimension
            repeated_observations_ood = state.repeat(
                self.ood_action_samples, 1, 1
            ).reshape(self.ood_action_samples * batch_size, -1)

            repeated_zs_ood = z_policy.repeat(self.ood_action_samples, 1, 1).reshape(
                self.ood_action_samples * batch_size, -1
            )
            ood_actions = torch.empty(
                size=(self.ood_action_samples * batch_size, action.shape[-1]),
                device=action.device,
            ).uniform_(-1, 1)

            if self.actor_action_samples > 0:
                repeated_observations_actor = state.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * batch_size, -1)

                repeated_next_observations_actor = next_state.repeat(
                    self.actor_action_samples, 1, 1
                ).reshape(self.actor_action_samples * batch_size, -1)

                repeated_zs_actor = z_policy.repeat(self.actor_action_samples, 1, 1).reshape(
                    self.actor_action_samples * batch_size, -1
                )

                ##
                actor_current_actions = self.target_calcute.act(state = repeated_observations_actor, \
                                                                z_policy = repeated_zs_actor, \
                                                                type = ActorValueType.SAMPLE)

                actor_next_actions = self.target_calcute.act(
                    state = repeated_next_observations_actor,
                    z_policy = repeated_zs_actor,
                    type = ActorValueType.SAMPLE)  # [actor_action_samples * batch_size, action_length]

        # get cml Fs
        ood_F = self.calcute.forward_representation(state = repeated_observations_ood, \
                                                             action = ood_actions, \
                                                             z_policy = repeated_zs_ood)
                                                            # [ood_action_samples * batch_size, latent_dim]

        F: torch.Tensor = items['F']

        if self.actor_action_samples > 0:
            actor_current_F = self.calcute.forward_representation(
                state = repeated_observations_actor, action = actor_current_actions, z_policy = repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]
            actor_next_F = self.calcute.forward_representation(
                state = repeated_next_observations_actor, action = actor_next_actions, z_policy = repeated_zs_actor
            )  # [actor_action_samples * batch_size, latent_dim]

            repeated_F = F.repeat(self.actor_action_samples, 1, 1).reshape(self.actor_action_samples * batch_size, -1)

            if F is [tuple, list] and len(F) == 2:
                cat_F = [torch.cat([ood_F[idx], actor_current_F[idx], actor_next_F[idx], repeated_F[idx]], dim=0) for idx in range(len(F))]
            else:
                if len(F.shape) == 3:
                    cat_F = torch.cat([ood_F, actor_current_F, actor_next_F, repeated_F], dim=1)
                else:
                    cat_F = torch.cat([ood_F, actor_current_F, actor_next_F, repeated_F], dim=0)
        else:
            cat_F = ood_F

        return cat_F


    def _calcute_mc_penalty(self, inputs: dict,
        items: dict, next_items: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the measure conservative penalty.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            next_observations: next observation tensor of shape
                                                        [batch_size, observation_length]
            zs: task tensor of shape [batch_size, z_dimension]
            actor_std_dev: standard deviation of the actor
            F1: forward embedding no. 1
            F2: forward embedding no. 2
            B_next: backward embedding
            M1_next: successor measure no. 1
            M2_next: successor measure no. 2
        Returns:
            conservative_penalty: the measure conservative penalty
            metrics: dictionary of metrics for logging
        """

        B_next: torch.Tensor = items['B']
        M_next: torch.Tensor = items['M']
        batch_size = B_next.shape[0]

        cat_F = self._calcute_penalty_F(inputs, items, next_items)

        if cat_F is [tuple, list] and len(cat_F) == 2:
            cml_cat_M = [torch.einsum("sd, td -> st", cat_Fi, B_next).reshape(self.total_action_samples, batch_size, -1) for cat_Fi in cat_F]
            cml_logsumexp = torch.sum([torch.logsumexp(cml_cat_Mi, dim=0).mean() for cml_cat_Mi in cml_cat_M])
            m_mean = torch.sum([M_nexti.mean() for M_nexti in M_next])
        else:
            if len(cat_F.shape) == 3:
                cml_cat_M = torch.einsum("psd, td -> pst", cat_F, B_next)
                cml_logsumexp = torch.logsumexp(cml_cat_M, dim=0).mean() * cat_F.shape[0]
                m_mean = M_next.mean() * cat_F.shape[0]
            else:
                cml_cat_M = torch.einsum("sd, td -> st", cat_F, B_next)
                cml_logsumexp = torch.logsumexp(cml_cat_M, dim=0).mean()
                m_mean = M_next.mean()

        conservative_penalty = cml_logsumexp - m_mean

        metrics = {
            "fb/cml_penalty": conservative_penalty.item(),
            "fb/cml_cat_M1": cml_cat_M[0].mean().item(),
            "fb/cml_cat_M2": cml_cat_M[1].mean().item(),
        }

        return conservative_penalty, metrics


    def _calcute_vc_penalty(self, inputs: dict,
            items: dict, next_items: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates the value conservative penalty. See section 3
        and appendix B.1.3.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            next_observations: next observation tensor of shape
                                                     [batch_size, observation_length]
            zs: task tensor of shape [batch_size, z_dimension]
            actor_std_dev: standard deviation of the actor
            F1: forward embedding no. 1
            F2: forward embedding no. 2
        Returns:
            conservative_penalty: the value conservative penalty
            metrics: dictionary of metrics for logging
        """

        B_next: torch.Tensor = items['B']
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量
        batch_size = B_next.shape[0]

        cat_F = self._calcute_penalty_F(inputs, items, next_items)

        repeated_zs = z_policy.repeat(self.total_action_samples, 1, 1).reshape(
            self.total_action_samples * batch_size, -1
        )

        Q = items["Q"]

        if cat_F is [tuple, list] and len(cat_F) == 2:
            cql_cat_Q = [torch.einsum("sd, sd -> s", cat_Fi, repeated_zs).reshape(self.total_action_samples, batch_size, -1) for cat_Fi in cat_F]
            cql_logsumexp = torch.sum([torch.logsumexp(cql_cat_Qi, dim=0).mean() for cql_cat_Qi in cql_cat_Q])

            q_mean = torch.sum([Qi.mean() for Qi in Q])
        else:
            if len(cat_F.shape) == 3:
                cql_cat_Q = torch.einsum("psd, sd -> ps", cat_F, repeated_zs)
                cql_logsumexp = torch.logsumexp(cql_cat_Q, dim=0).mean() * cql_cat_Q.shape[0]
                q_mean = Q.mean() * cql_cat_Q.shape[0]
            else:
                cql_cat_Q = torch.einsum("sd, sd -> s", cat_F, repeated_zs)
                cql_logsumexp = torch.logsumexp(cql_cat_Q, dim=0).mean()
                q_mean = Q.mean()

        conservative_penalty = cql_logsumexp - q_mean

        metrics = {
            "fb/cql_penalty": conservative_penalty.item(),
            "fb/cql_cat_Q1": cql_cat_Q[0].mean().item(),
            "fb/cql_cat_Q2": cql_cat_Q[1].mean().item(),
        }

        return conservative_penalty, metrics


    def _calcute_penalty_loss(self, inputs: dict,
            items: dict, next_items: dict) -> Tuple[torch.Tensor, Dict[str, float]]:

        total_penalty = 0
        metrics = {}
        if self.config.if_measure_conservative_penalty:
            penalty, metrics_items = self._calcute_mc_penalty(inputs, items, next_items)
            total_penalty = penalty
            metrics.update(metrics_items)

        if self.config.if_value_conservative_penalty:
            penalty, metrics_items = self._calcute_vc_penalty(inputs, items, next_items)
            total_penalty += penalty
            metrics.update(metrics_items)

        # alpha auto-tuning
        if False: #self.lagrange: #
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6)
            alpha_loss = (
                -0.5 * alpha * (penalty - self.config.conservative_target_conservative_penalty)
            )

            self.critic_alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_optimizer.step()
            alpha = torch.clamp(self.critic_log_alpha.exp(), min=0.0, max=1e6).detach()
            alpha_loss = alpha_loss.detach().item()

            metrics = {
                "train/alpha": alpha,
                "train/alpha_loss": alpha_loss,
            }

        # fixed alpha
        else:
            alpha = self.config.conservative_alpha

        losses = alpha * total_penalty
        return losses, metrics

    def _calcute_loss(self, inputs: dict, step: int) -> dict:
        discount: torch.Tensor = inputs['discount']  # 折扣因子

        metrics: Dict[str, float] = {}  # 训练指标字典

        # 计算后继度量
        next_items = self._calcute_next(inputs, step)  # 下一状态相关量
        items = self._calcute_curent(inputs, step)  # 当前状态相关量

        # FB损失：后继度量学习损失
        fb_loss, metrics_items = self._calcute_fb_loss(items, next_items, discount)
        loss = fb_loss.clone()  # 总损失初始化为FB损失
        metrics.update(metrics_items)

        # Q损失：如果配置了Q值计算
        if self.config.if_calcute_q:
            q_loss, metrics_items = self._calcute_q_loss(inputs, items, next_items, discount)
            loss += q_loss * self.config.q_loss_coef  # 加权Q损失
            metrics.update(metrics_items)

        if self.config.if_measure_conservative_penalty or self.config.if_value_conservative_penalty:
            penalty_loss, metrics_items = self._calcute_penalty_loss(self, inputs, items, next_items)
            loss += penalty_loss
            metrics.update(metrics_items)

        # 后向表示的正交性损失
        orth_loss, metrics_items = self._calcute_orth_loss(items)
        loss += self.config.ortho_coef * orth_loss  # 加权正交性损失
        metrics.update(metrics_items)

        with torch.no_grad():
            # metrics["fb/z_norm"] = torch.norm(inputs['z_policy'].detach(), dim=-1).mean()
            ##
            metrics["fb/loss"] = loss.detach()

        return loss, metrics
