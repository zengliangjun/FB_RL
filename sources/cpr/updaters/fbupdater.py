from base import updaters, models

from cpr.updaters import configs
from cpr.models import fb
from utils.actor_post import ActorValueType


from typing import Dict, Union
import torch
from torch.nn import functional as F
from torch.optim import Optimizer


class FBUpdater(updaters.Updater):

    config: configs.FBConfig

    calcute: fb.FBCalcute
    target_calcute: fb.FBCalcute

    forward_map_optim: Optimizer
    backward_map_optim: Optimizer

    off_diag: torch.Tensor
    off_diag_sum: float

    def __init__(self, cfg: configs.FBConfig, model: models.BaseModel):
        super(FBUpdater, self).__init__(cfg, model)

        self.calcute = model.calcute
        self.target_calcute = model.target_calcute

        self.forward_map_optim = getattr(model, "forward_map_optim")
        self.backward_map_optim = getattr(model, "backward_map_optim")

        self.off_diag = None
        self.off_diag_sum = 0

    @torch.no_grad()
    def _calcute_next(self, inputs: dict, step: int):
        next_state: torch.Tensor = inputs['next_state']  # 下一状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        next_action = self.target_calcute.act(state = next_state, z_policy = z_policy, type = ActorValueType.SAMPLE, step = step)

        # 使用目标网络计算下一状态的前向和后向表示
        next_F = self.target_calcute.forward_representation(state = next_state, action = next_action, z_policy = z_policy)  # batch x z_dim
        next_B = self.target_calcute.backward_representation(state = next_state)  # batch x z_dim

        items = {}

        # 处理双网络情况（如TD3）
        if next_F is [tuple, list] and len(next_F) == 2:
            # 分别计算两个网络的后继度量矩阵
            next_M1, next_M2 = [torch.einsum('sd, td -> st', next_Fi, next_B) for next_Fi in next_F]  # batch x batch
            next_M = torch.min(next_M1, next_M2)  # 取最小值避免过度乐观
            items["M"] = next_M

            # 如果配置了Q值计算，计算下一状态的Q值
            if self.config.if_calcute_q:
                next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', next_Fi, z_policy) for next_Fi in next_F]
                next_Q = torch.min(next_Q1, nextQ2)  # 双Q学习
                items["Q"] = next_Q

        else:
            # 处理集成网络情况
            if len(next_F.shape) == 3:
                # 集成网络：多个并行网络计算后继度量
                next_M = torch.einsum('psd, td -> pst', next_F, next_B)  # p x batch x batch

                # 使用不确定性估计计算最终后继度量
                assert hasattr(self.target_calcute, 'get_targets_uncertainty')
                # 参考：metamotivo/metamotivo/fb/agent.py : update_fb
                next_M = self.target_calcute.get_targets_uncertainty(
                    next_M, pessimism_penalty=self.config.pessimism_penalty)
                items["M"] = next_M

                # 如果配置了Q值计算，计算下一状态的Q值
                if self.config.if_calcute_q:
                    next_Q = torch.einsum('psd, sd -> ps', next_F, z_policy)
                    next_Q = self.target_calcute.get_targets_uncertainty(
                        next_Q, pessimism_penalty=self.config.pessimism_penalty)
                    items["Q"] = next_Q
            else:
                # 单网络情况：直接计算后继度量
                next_M = torch.einsum('sd, td -> st', next_F, next_B)  # batch x batch
                items["M"] = next_M

                # 如果配置了Q值计算，计算下一状态的Q值
                if self.config.if_calcute_q:
                    next_Q = torch.einsum('sd, sd -> s', next_F, z_policy)
                    items["Q"] = next_Q

        return items

    def _calcute_curent(self, inputs: dict, step: int):
        """计算当前状态的相关量

        使用在线网络计算当前状态的前向和后向表示。

        Args:
            inputs: 输入数据字典
            step: 当前训练步数

        Returns:
            items: 包含当前状态相关量的字典
        """
        state: torch.Tensor = inputs['state']  # 当前状态
        action: torch.Tensor = inputs['action']  # 当前动作
        next_state: torch.Tensor = inputs['next_state']  # 下一状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        # 使用在线网络计算当前状态的前向和后向表示
        F = self.calcute.forward_representation(state = state, action = action, z_policy = z_policy)
        B = self.calcute.backward_representation(state = next_state)

        items = {"B": B, "F": F}  # 包含后向表示的字典

        # 处理双网络情况
        if F is [tuple, list] and len(F) == 2:
            # 分别计算两个网络的后继度量矩阵
            M = [torch.einsum('sd, td -> st', Fi, B) for Fi in F]
            items["M"] = M

            # 如果配置了Q值计算，计算当前状态的Q值
            if True: #self.config.if_calcute_q:
                Q = [torch.einsum('sd, sd -> s', Fi, z_policy) for Fi in F]
                items["Q"] = Q

        else:
            # 处理集成网络情况
            if len(F.shape) == 3:
                # 集成网络：多个并行网络计算后继度量
                M = torch.einsum('psd, td -> pst', F, B)  # p x batch x batch
                items["M"] = M                            # p x batch x batch

                # 如果配置了Q值计算，计算当前状态的Q值
                if True: #self.config.if_calcute_q:
                    Q = torch.einsum('psd, sd -> ps', F, z_policy)
                    items["Q"] = Q
            else:
                # 单网络情况：直接计算后继度量
                M = torch.einsum('sd, td -> st', F, B)  # batch x batch
                items["M"] = M

                # 如果配置了Q值计算，计算当前状态的Q值
                if True: # self.config.if_calcute_q:
                    Q = torch.einsum('sd, sd -> s', F, z_policy)
                    items["Q"] = Q

        return items


    def _calcute_fb_loss(self, M: torch.Tensor, next_M: torch.Tensor, discount: Union[float, torch.Tensor]):
        """计算前向-后向损失

        计算后继度量矩阵的损失，包括非对角线损失和对角线损失。

        Args:
            M: 当前状态的后继度量矩阵
            next_M: 下一状态的后继度量矩阵
            discount: 折扣因子

        Returns:
            fb_loss: 前向-后向损失值
            metrics: 损失指标字典
        """
        # 延迟初始化非对角线掩码
        if self.off_diag is None:
            I = torch.eye(*next_M.size(), device=next_M.device)  # 单位矩阵
            self.off_diag = ~I.bool()  # 非对角线掩码
            self.off_diag_sum = self.off_diag.sum().item()  # 非对角线元素数量

        ## TODO: fb_diag
        # 非对角线损失：后继度量矩阵的非对角线元素应该满足贝尔曼方程
        if isinstance(M, Union[list, tuple]) and len(M) == 2:
            # 双网络情况，计算每个网络的非对角线损失并求和
            diff = (Mi - discount * next_M for Mi in M)

            fb_offdiag = 0.5 * sum(diff_i[self.off_diag].pow(2).mean() for diff_i in diff)
            # 对角线损失：后继度量矩阵的对角线元素应该最大化（表示状态自身的可达性）
            # fb_diag = -sum(Mi.diag().mean() for Mi in M)
            fb_diag = sum(diff_i.diag().mean() for diff_i in diff)
        else:
            # 处理集成网络情况
            if len(M.shape) == 3:
                #  metamotivo/metamotivo/fb/agent.py  update_fb
                discount = discount[None, ...]
                off_diag = self.off_diag.expand(M.shape[0], -1, -1)  # num_parallel x batch x batch

                diff = M - discount * next_M  # num_parallel x batch x batch
                fb_offdiag = 0.5 * (diff[off_diag].pow(2).mean()) * M.shape[0]

                # fb_diag = - torch.diagonal(M, dim1=1, dim2=2).mean() * M.shape[0]
                fb_diag = torch.diagonal(diff, dim1=1, dim2=2).mean() * M.shape[0]
            else:
                # 单网络情况
                diff = M - discount * next_M

                fb_offdiag = 0.5 * (diff[self.off_diag].pow(2).mean())
                # fb_diag = - (M.diag().mean())
                fb_diag = (diff.diag().mean())

        # 总前向-后向损失
        fb_loss = fb_offdiag - fb_diag

        with torch.no_grad():
            items = {
                    "fb/fb_diag": fb_diag.detach(),
                    "fb/fb_offdiag": fb_offdiag.detach(),
                    "fb/fb_loss": fb_loss.detach()
                    }
        return fb_loss, items

    def _calcute_q_loss(self, inputs, items, next_items: torch.Tensor, discount: Union[float, torch.Tensor]):
        with torch.no_grad():
            # 计算隐式奖励
            B = items['B']  # 当前状态的后向表示
            z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

            ## calculate implicit reward
            cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
            inv_cov = torch.inverse(cov)  # z_dim x z_dim
            implicit_reward = (torch.matmul(B, inv_cov) * z_policy).sum(dim=-1)  # batch

            ## calculate target Q value
            if len(discount.shape) == 2:
                discount = discount.squeeze(-1)

            target_Q = implicit_reward.detach() + discount * next_items['Q']  # batch

        Q = items['Q']  # 当前状态的Q值
        if isinstance(Q, Union[list, tuple]) and len(Q) == 2:
            q_loss = 0.5 * sum([F.mse_loss(Qi, target_Q) for Qi in Q])
        else:
            if len(Q.shape) == 2:
                target_Q = target_Q.expand(Q.shape[0], -1)
                q_loss = 0.5 * F.mse_loss(Q, target_Q) * Q.shape[0]
            else:
                q_loss = 0.5 * F.mse_loss(Q, target_Q)

        with torch.no_grad():
            items = {
                "fb/target_Q": target_Q.mean().detach(),
                "fb/Q": Q.mean().detach(),
                "fb/q_loss": q_loss.detach()
            }
        return q_loss, items

    def _calcute_orth_loss(self, B: torch.Tensor):
        """计算正交性损失

        计算后向表示的正交性损失，确保技能向量的正交性。

        Args:
            B: 后向表示矩阵，形状为 [batch_size, z_dimension]

        Returns:
            orth_loss: 正交性损失
            metrics: 损失指标字典
        """
        # 延迟初始化非对角线掩码
        if self.off_diag is None:
            I = torch.eye((B.shape[0], B.shape[0]), device=B.device)  # 单位矩阵
            self.off_diag = ~I.bool()  # 非对角线掩码
            self.off_diag_sum = self.off_diag.sum().item()  # 非对角线元素数量

        # 计算后向表示的协方差矩阵
        Cov = torch.matmul(B, B.T)  # batch_size x batch_size

        # 对角线损失：协方差矩阵的对角线元素应该最大化（表示技能向量的强度）
        orth_loss_diag = - Cov.diag().mean()

        # 非对角线损失：协方差矩阵的非对角线元素应该最小化（表示技能向量的正交性）
        orth_loss_offdiag = 0.5 * Cov[self.off_diag].pow(2).mean()

        # 总正交性损失
        orth_loss = orth_loss_offdiag + orth_loss_diag

        with torch.no_grad():
            items = {
                "fb/orth_loss": orth_loss.detach(),
                "fb/orth_loss_diag": orth_loss_diag.detach(),
                "fb/orth_loss_offdiag": orth_loss_offdiag.detach(),

            }
        return orth_loss, items


    def _calcute_loss(self, inputs: dict, step: int) -> dict:
        discount: torch.Tensor = inputs['discount']  # 折扣因子

        metrics: Dict[str, float] = {}  # 训练指标字典

        # 计算后继度量
        next_items = self._calcute_next(inputs, step)  # 下一状态相关量
        items = self._calcute_curent(inputs, step)  # 当前状态相关量

        # FB损失：后继度量学习损失
        fb_loss, metrics_items = self._calcute_fb_loss(items['M'], next_items['M'], discount)
        loss = fb_loss.clone()  # 总损失初始化为FB损失
        metrics.update(metrics_items)

        # Q损失：如果配置了Q值计算
        if self.config.if_calcute_q:
            q_loss, metrics_items = self._calcute_q_loss(inputs, items, next_items, discount)
            loss += q_loss * self.config.q_loss_coef  # 加权Q损失
            metrics.update(metrics_items)

        # 后向表示的正交性损失
        orth_loss, metrics_items = self._calcute_orth_loss(items['B'])
        loss += self.config.ortho_coef * orth_loss  # 加权正交性损失
        metrics.update(metrics_items)

        with torch.no_grad():
            metrics["fb/target_M"] = next_items['M'].detach().mean()
            if isinstance(items['M'], Union[list, tuple]) and len(items['M']) == 2:
                metrics["fb/M1"] = items['M'][0].detach().mean()
                metrics["fb/F1"] = items['F'][0].detach().mean()
            else:
                metrics["fb/M1"] = items['M'][0].detach().mean()
                metrics["fb/F1"] = items['F'][0].detach().mean()

            metrics["fb/B"] = items['B'].detach().mean()
            # metrics["fb/Q"] = items['Q'].detach().mean().item()

            metrics["fb/B_norm"] = torch.norm(items['B'].detach(), dim=-1).mean()
            metrics["fb/z_norm"] = torch.norm(inputs['z_policy'].detach(), dim=-1).mean()
            ##
            metrics["fb/loss"] = loss.detach()

        return loss, metrics

    def _calcute_loss_org(self, inputs: dict, step: int) -> dict:
        next_state: torch.Tensor = inputs['next_state']  # 下一状态
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        with torch.no_grad():
            next_action = self.target_calcute.act(state = next_state, z_policy = z_policy, type = ActorValueType.SAMPLE, step = step)

            target_Fs = self.target_calcute.forward_representation(state = next_state, action = next_action, z_policy = z_policy)  # num_parallel x batch x z_dim
            target_B = self.target_calcute.backward_representation(state = next_state)  # batch x z_dim
            target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
            target_M = self.target_calcute.get_targets_uncertainty(
                    target_Ms, pessimism_penalty=self.config.pessimism_penalty)

        state: torch.Tensor = inputs['state']  # 当前状态
        action: torch.Tensor = inputs['action']  # 当前动作
        z_policy: torch.Tensor = inputs['z_policy']  # 技能向量

        # compute FB loss
        Fs = self.calcute.forward_representation(state, action, z_policy)  # num_parallel x batch x z_dim
        B = self.calcute.backward_representation(next_state)  # batch x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        discount: torch.Tensor = inputs['discount']  # 折扣因子

        if self.off_diag is None:
            I = torch.eye(*target_M.size(), device=target_M.device)  # 单位矩阵
            self.off_diag = ~I.bool()  # 非对角线掩码
            self.off_diag_sum = self.off_diag.sum().item()  # 非对角线元素数量

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.config.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z_policy.device, dtype=z_policy.dtype)
        if self.config.if_calcute_q:
            with torch.no_grad():
                next_Qs = (target_Fs * z_policy).sum(dim=-1)  # num_parallel x batch
                next_Q = self.target_calcute.get_targets_uncertainty(next_Qs, self.config.pessimism_penalty)  # batch

                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z_policy).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch

                expanded_targets = target_Q.expand(Fs.shape[0], -1)

            Qs = (Fs * z_policy).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += self.config.q_loss_coef * q_loss

        with torch.no_grad():
            output_metrics = {
                "fb/target_M": target_M.mean().detach(),
                "fb/M1": Ms[0].mean().detach(),
                "fb/F1": Fs[0].mean().detach(),
                "fb/B": B.mean().detach(),
                "fb/B_norm": torch.norm(B, dim=-1).mean().detach(),
                "fb/z_norm": torch.norm(z_policy, dim=-1).mean().detach(),

                "fb/loss": fb_loss.detach(),

                "fb/fb_loss": (fb_diag + fb_offdiag).detach(),
                "fb/fb_diag": fb_diag.detach(),
                "fb/fb_offdiag": fb_offdiag.detach(),

                "fb/orth_loss": orth_loss.detach(),
                "fb/orth_loss_diag": orth_loss_diag.detach(),
                "fb/orth_loss_offdiag": orth_loss_offdiag.detach(),

                "fb/q_loss": q_loss.detach(),
                "fb/target_Q": target_Q.mean().detach(),
                "fb/Q": Qs.mean().detach(),
            }
        return fb_loss, output_metrics

    def update(self, batch: Dict, step: int) -> Dict[str, torch.Tensor]:

        loss, metrics = self._calcute_loss(batch, step)

        # 优化FB网络
        self.forward_map_optim.zero_grad(set_to_none=True)
        self.backward_map_optim.zero_grad(set_to_none=True)

        loss.backward()

        if self.config.clip_grad_norm is not None \
            and self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.forward_map.parameters(), self.config.clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.model.backward_map.parameters(), self.config.clip_grad_norm
            )

        self.forward_map_optim.step()
        self.backward_map_optim.step()

        return metrics

    def save_dict(self, collect_dict: dict, prefix: str):
        collect_dict[f"{prefix}_forward_map_optim"] = self.forward_map_optim.state_dict()
        collect_dict[f"{prefix}_backward_map_optim"] = self.backward_map_optim.state_dict()

    def resume_dict(self, collect_dict: dict, prefix: str):
        self.forward_map_optim.load_state_dict(collect_dict[f"{prefix}_forward_map_optim"])
        self.backward_map_optim.load_state_dict(collect_dict[f"{prefix}_backward_map_optim"])
