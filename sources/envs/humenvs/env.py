import torch
import numpy as np
from typing import Dict, Union, Tuple, Mapping
from humenv import make_humenv
import gymnasium.wrappers as wrappers
import signal
import sys
import traceback
from loguru import logger as ulogger

from envs.humenvs import configs

class HumenvWarp:

    def __init__(self, cfg: configs.HumenvConfig):
        self.config = cfg

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        try:
            ulogger.info(f"Creating humenv with {self.config.num_envs} parallel environments")
            train_env, mp_info = make_humenv(
                num_envs=self.config.num_envs,
                # vectorization_mode="sync",
                wrappers=[
                    wrappers.FlattenObservation,
                    lambda env: wrappers.TimeAwareObservation(env, flatten=False),
                ],
                render_width=320,
                render_height=320,
                motions=self.config.envinit_motions,
                motion_base_path=self.config.envinit_motions_root,
                fall_prob=0.2,
                state_init="MoCapAndFall",
            )
            self.envs = train_env
            self.mp_info = mp_info
            ulogger.info("Successfully created humenv environments")
        except Exception as e:
            ulogger.error(f"Failed to create humenv environments: {e}")
            ulogger.error(traceback.format_exc())
            raise

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            ulogger.info(f"Received signal {signum}, shutting down environments gracefully")
            self.end()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def __del__(self):
        try:
            self.end()
        except:
            pass

    def _recursive_to_tensor(self, _s, _d):
        for k, v in _s.items():
            if isinstance(v, Mapping):
                _dv = {}
                self._recursive_to_tensor(v, _dv)
                _d[k]= _dv
            else:
                _d[k]= torch.tensor(v)


    def reset(self) -> Tuple[torch.Tensor, Dict[str, object]]:
        # obs, extras
        td, infos = self.envs.reset()
        obs = torch.tensor(td["obs"], dtype=torch.float32)

        self.step_count = torch.tensor(td["time"])

        tensor_infos = {}
        self._recursive_to_tensor(infos, tensor_infos)
        return obs, tensor_infos

    def timestep(self) -> torch.Tensor:
        return self.step_count

    def sample_action(self) -> torch.Tensor:
        actions = self.envs.action_space.sample().astype(np.float32)
        return torch.tensor(actions)

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> \
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor, Dict[str, object]]:

        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy()

        td, rewards, terminated, truncated, infos = self.envs.step(action)
        dones = np.logical_or(terminated.ravel(), truncated.ravel())

        obs = torch.tensor(td["obs"], dtype=torch.float32)
        self.step_count = torch.tensor(td["time"])

        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        tensor_infos = {}

        infos["terminated"] = terminated[:, None]
        infos["truncated"] = truncated[:, None]
        self._recursive_to_tensor(infos, tensor_infos)
        return obs, rewards, dones, tensor_infos

    def end(self):
        try:
            self.envs.close()
        except:
            pass

    def update_priorities(self, items):
        motions_id, priorities, idxs = items
        if self.mp_info is not None:
            self.mp_info["motion_buffer"].update_priorities(motions_id=motions_id, priorities=priorities.cpu().numpy())
        else:
            self.envs.unwrapped.motion_buffer.update_priorities(motions_id=motions_id, priorities=priorities.cpu().numpy())
