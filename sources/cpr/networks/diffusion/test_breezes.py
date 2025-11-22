import os
import sys
import os.path as osp
import dataclasses
import torch

script_dir = osp.abspath(osp.dirname(__file__))
workroot_dir = osp.abspath(osp.join(script_dir, "../../../.."))
os.chdir(workroot_dir)

source_dir = f"{workroot_dir}/sources"
if source_dir not in sys.path:
    sys.path.insert(0, source_dir)


from cpr.networks import configs
from base import networks

@dataclasses.dataclass
class BackwardMapConfig(configs.BreezeBackwardMapConfig):
    device: str = "cpu"
    state_dimension: int = 24
    action_dimension: int = 29
    z_dimension: int = 50
    # out_dimension: int = 50

    hidden_dimension: int = 2048
    hidden_layers: int = 2

    d_model: int = 256
    nhead: int = 8
    dropout: float = 0.1

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    target_tau: float = 0.01


@dataclasses.dataclass
class ForwardMapConfig(configs.BreezeForwardMapConfig):
    device: str = "cpu"

    num_parallel: int = 2

    state_dimension: int = 24
    action_dimension: int = 29
    z_dimension: int = 50

    hidden_dimension: int = 2048
    embedding_layers: int = 2
    hidden_layers: int = 2

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    target_tau: float = 0.01

from cpr.networks.diffusion import configs as dconfigs

@dataclasses.dataclass
class ActorDiffusionConfig(dconfigs.ActorDiffusionConfig):
    device: str = "cpu"

    state_dimension: int = 24
    action_dimension: int = 29
    z_dimension: int = 50

    embedding_simple: bool = True
    hidden_dimension: int = 256
    embedding_layers: int = 2

    diffusion_block: dconfigs.IDQLConfig = dconfigs.IDQLConfig(
                            hidden_dimension = 256,
                            hidden_layers = 2
                        )

    optim_lr: float = 1e-4  # 网络的学习率
    optim_weight_decay: float = 0

    target_tau: float = 0.01

if __name__ == "__main__":

    if False:
        cfg = BackwardMapConfig()
        print(cfg.to_json())

        back = networks.NetProxy(cfg)
        print(back)

        _state = torch.randn((4, 24), dtype = torch.float32)
        _out = back(_state)
        print(_out)


        _state = torch.randn((4, 24), dtype = torch.float32)
        _action = torch.randn((4, 29), dtype = torch.float32)
        _z_policy = torch.randn((4, 50), dtype = torch.float32)

        cfg = ForwardMapConfig()
        print(cfg.to_json())

        forward = networks.NetProxy(cfg)
        print(forward)

        _out = forward(_state, _action, _z_policy)
        print(_out.shape)
        diff = _out[0] - _out[1]
        print(diff)
        print(diff.mean())


    _state = torch.randn((1, 24), dtype = torch.float32)
    _action = torch.randn((1, 29), dtype = torch.float32)
    _z_policy = torch.randn((1, 50), dtype = torch.float32)

    cfg = ActorDiffusionConfig()
    actor = networks.NetProxy(cfg)

    action = actor.get_action(
        state = _state,
        z = _z_policy,
        num = 10)

    print(action.shape)

    _state = torch.randn((4, 24), dtype = torch.float32)
    _action = torch.randn((4, 29), dtype = torch.float32)
    _z_policy = torch.randn((4, 50), dtype = torch.float32)


    action = actor.get_action(
        state = _state,
        z = _z_policy,
        num = 10)

    print(action.shape)


    _state = torch.randn((4, 24), dtype = torch.float32)
    _action = torch.randn((4, 29), dtype = torch.float32)
    _z_policy = torch.randn((4, 50), dtype = torch.float32)


    loss = actor.policy_loss(
        action = _action,
        state = _state,
        z_policy = _z_policy)

    print(loss)
