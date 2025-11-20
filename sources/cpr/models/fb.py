from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

from base import models
from cpr.models import configs
from cpr.networks import utils

class FBCalcute(models.BaseModelCalcute):

    def __init__(self, cfg: models.BaseModelConfig):
        super(FBCalcute, self).__init__(cfg)

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> torch.Tensor:
        """计算目标值和不确定性

        基于并行网络的预测结果计算均值、不确定性和悲观估计。
        用于离线强化学习中的不确定性惩罚。

        Args:
            preds: 并行网络的预测结果，形状为 [num_parallel, ...]
            pessimism_penalty: 悲观惩罚系数

        Returns:
            preds_mean: 预测均值
            preds_unc: 预测不确定性
            pessimistic_target: 悲观目标值（均值 - 惩罚 * 不确定性）
        """
        dim = 0  # 并行维度
        preds_mean = preds.mean(dim=dim)  # 计算均值

        # 计算所有网络对之间的差异
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...

        # 计算不确定性（所有网络对差异的平均值）
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]  # 网络对数量
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),  # 在并行维度上求和
            )
            / num_parallel_scaling  # 除以网络对数量得到平均值
        )

        # 计算悲观目标值
        # preds_mean, preds_unc,
        return preds_mean - pessimism_penalty * preds_unc


    def reward_inference(self, state: torch.Tensor, \
                         reward: torch.Tensor, \
                         inference_batch_size: int = 500_000,
                         weight: torch.Tensor | None = None) -> torch.Tensor:

        state = self.state_prepose(state)

        num_batches = int(np.ceil(state.shape[0] / inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * inference_batch_size, (i + 1) * inference_batch_size
            B = self.backward_representation(state[start_idx:end_idx].to(self.config.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.config.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, state: torch.Tensor,
                                  reward: torch.Tensor,
                                  inference_batch_size: int = 500_000) -> torch.Tensor:

        # state = self.state_prepose(state)

        return self.reward_inference(state = state,
                                     reward = reward,
                                     inference_batch_size = inference_batch_size,
                                    weight = F.softmax(10 * reward, dim=0))

    def goal_inference(self, state: torch.Tensor) -> torch.Tensor:
        state = self.state_prepose(state)

        return self.backward_representation(state.to(self.config.device))

    def tracking_inference(self, state: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "seq_length")

        state = self.state_prepose(state)

        z = self.backward_representation(state.to(self.config.device))
        for step in range(z.shape[0]):
            end_idx = min(step + self.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)

class  FBModel(models.BaseModel):

    def __init__(self, cfg: configs.FBModelConfig):
        super(FBModel, self).__init__(cfg)


    def _init_network(self):
        super(FBModel, self)._init_network()
        if not hasattr(self, "state_prepose"):
            self.state_prepose = nn.BatchNorm1d(self.config.state_dimension, affine=False, momentum=0.01)

            setattr(self.calcute, "state_prepose", self.state_prepose)
            setattr(self.target_calcute, "state_prepose", self.state_prepose)

        self.apply(utils.weight_init)
