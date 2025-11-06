from tqdm import tqdm
from humenv.misc.motionlib import canonicalize, load_episode_based_h5
import numpy as np
import torch
from typing import Union, Mapping
from pathlib import Path
from loguru import logger as ulogger

from cpr.buffers import configs, org_buffer

def load_expert_trajectories(motions: str, motions_root: str, device: str, sequence_length: int) -> org_buffer.TrajectoryBuffer:
    with open(motions, "r") as txtf:
        h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
    episodes = []
    for h5 in tqdm(h5files, leave=False):
        try:
            h5 = canonicalize(h5, base_path=motions_root)
        except:
            continue

        _ep = load_episode_based_h5(h5, keys=None)
        for el in _ep:
            el["observation"] = el["observation"].astype(np.float32)
            del el["file_name"]
        episodes.extend(_ep)
    buffer = org_buffer.TrajectoryBuffer(
        capacity=len(episodes),
        seq_length=sequence_length,
        device=device,
    )
    buffer.extend(episodes)
    return buffer

class MotionBuffer:
    def __init__(self, cfg: configs.MotionBufferConfig):
        self.config = cfg
        self.device = torch.device("cpu")

        self.buffer = load_expert_trajectories( \
            cfg.motions,
            cfg.motions_root,
            self.device,
            cfg.seq_length)


    def __len__(self) -> int:
        return len(self.buffer)

    def empty(self) -> bool:
        return self.buffer.empty()

    def extend(self, data: torch.Tensor) -> None:
        raise Exception("Don\'t support extend")

    def sample(self, num, device: Union[torch.device, str]=None) -> torch.Tensor:
        def recursive_to_device(_s, _d):
            for k, v in _s.items():
                if isinstance(v, Mapping):
                    _dv = {}
                    recursive_to_device(v, _dv)
                    _d[k]= _dv
                else:
                    _d[k]= v.to(device)

        if device is None:
            device = self.config.device

        items = self.buffer.sample(num)

        results = {}
        recursive_to_device(items, results)
        return results

    def _index_in_buffer(self):
        index_in_buffer = {}
        for i, ep in enumerate(self.buffer.storage):
            index_in_buffer[ep["motion_id"][0].item()] = i
        return index_in_buffer

    def priorities(self, metrics):
        index_in_buffer = self._index_in_buffer()

        motions_id, priorities, idxs = [], [], []
        for _, metr in metrics.items():
            motions_id.append(metr["motion_id"])
            priorities.append(metr["emd"])
            idxs.append(index_in_buffer[metr["motion_id"]])
        priorities = (
            torch.clamp(
                torch.tensor(priorities, dtype=torch.float32, device=self.device),
                min=self.config.prioritization_min_val,
                max=self.config.prioritization_max_val,
            )
            * self.config.prioritization_scale
        )
        bins = torch.floor(priorities)
        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
            mask = bins == i
            n = mask.sum().item()
            if n > 0:
                priorities[mask] = 1 / n

        return motions_id, priorities, idxs

    def update_priorities(self, items):
        motions_id, priorities, idxs = items
        self.buffer.update_priorities(
            priorities=priorities.to(self.device),
            idxs=torch.tensor(np.array(idxs), device=self.device)
        )


class DictBuffer:
    def __init__(self, cfg: configs.DictBufferConfig):
        self.config = cfg

        self.buffer = org_buffer.DictBuffer( \
            cfg.capacity,
            torch.device("cpu"))


    def __len__(self) -> int:
        return len(self.buffer)

    def empty(self) -> bool:
        return self.buffer.empty()

    def extend(self, data: dict) -> None:
        self.buffer.extend(data)

    def sample(self, num, device: Union[torch.device, str]=None) -> dict:
        items = self.buffer.sample(num)
        if device is None:
            device = self.config.device

        for key, value in items.items():
            if isinstance(value, torch.Tensor):
                items[key] = value.to(device)

        return items

    def get_full_buffer(self):
        return self.buffer.get_full_buffer()


def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    ulogger.info(f"Data path: {path}")
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    files = list(path.glob("*.npz"))
    if -1 == num_episodes:
        num_episodes = len(files)
    else:
        num_episodes = min(num_episodes, len(files))

    for i in tqdm(range(num_episodes)):
        f = files[i]
        data = np.load(str(f))
        storage["observation"].append(data["observation"][:-1].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["observation"].append(data["observation"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=np.bool_))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    for k in storage:
        if k == "next":
            for k1 in storage[k]:
                storage[k][k1] = np.concatenate(storage[k][k1])
        else:
            storage[k] = np.concatenate(storage[k])
    return storage



class DMCBuffer:
    def __init__(self, cfg: configs.DMCBufferConfig):
        self.config = cfg

        data = load_data(
            self.config.dataset_root,
            self.config.dataset_expl_agent,
            self.config.domain_name,
            self.config.load_n_episodes,
        )


        self.buffer = org_buffer.DictBuffer( \
            data["observation"].shape[0],
            torch.device("cpu"))

        self.buffer.extend(data)
        del data

    def sample(self, num, device: Union[torch.device, str]=None) -> dict:
        items = self.buffer.sample(num)
        if device is None:
            device = self.config.device

        for key, value in items.items():
            if isinstance(value, torch.Tensor):
                items[key] = value.to(device)

        return items
