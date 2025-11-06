
import torch
import numpy as np

from base_envs.simple_buffer import configs

class ZBuffer:
    def __init__(self, cfg: configs.BufferConfig):
        self.config = cfg
        self.storage = torch.zeros((cfg.capacity, cfg.dim), dtype=torch.float32)
        self.idx = 0
        self.is_full = False

    def __len__(self) -> int:
        return self.config.capacity if self.is_full else self.idx

    def empty(self) -> bool:
        return self.idx == 0 and not self.is_full

    def extend(self, data: torch.Tensor) -> None:
        data = data.detach().cpu()
        if self.idx + data.shape[0] >= self.config.capacity:
            diff = self.config.capacity - self.idx
            self.storage[self.idx : self.idx + data.shape[0]] = data[:diff]
            self.storage[: data.shape[0] - diff] = data[diff:]
            self.is_full = True
        else:
            self.storage[self.idx : self.idx + data.shape[0]] = data
        self.idx = (self.idx + data.shape[0]) % self.config.capacity

    def sample(self, num, device=None) -> torch.Tensor:
        idx = np.random.randint(0, len(self), size=num)
        if device is None:
            device=self.config.device

        return self.storage[idx].clone().to(device)

    def load(self, file_name):
        items = torch.load(file_name)
        dim = items['dim']
        capacity = items['capacity']
        idx = items['idx']
        is_full = items['is_full']
        storage = items['storage']

        assert capacity == self.config.capacity
        assert dim == self.config.dim
        self.storage[...] = storage

        self.is_full = is_full
        self.idx = idx

    def save(self, file_name):
        items = {
            'dim': self.config.dim,
            'capacity': self.config.capacity,
            'idx': self.idx,
            'is_full': self.is_full,
            'storage': self.storage,
        }
        torch.save(items, file_name)
