# -*- coding: utf-8 -*-
from typing import Optional, List, Callable, Tuple

from ts_benchmark.data_loader.data_pool import Singleton
from ts_benchmark.utils.parallel.base import TaskResult
from ts_benchmark.utils.parallel.ray_backend import RayBackend
from ts_benchmark.utils.parallel.sequential_backend import SequentialBackend


__all__ = ["ParallelBackend"]


class ParallelBackend(metaclass=Singleton):

    #: all available backends
    BACKEND_DICT = {
        "ray": RayBackend,
        "sequential": SequentialBackend,
    }

    def __init__(self):
        self.backend = None
        self.default_timeout = None

    def init(
        self,
        backend: str = "ray",
        n_workers: Optional[int] = None,
        n_cpus: Optional[int] = None,
        gpu_devices: Optional[List[int]] = None,
        default_timeout: float = -1,
    ):
        if backend not in self.BACKEND_DICT:
            raise ValueError(f"Unknown backend name {backend}")
        if self.backend is not None:
            raise RuntimeError("Please close the backend before re-initializing")
        self.backend = self.BACKEND_DICT[backend](n_workers=n_workers, n_cpus=n_cpus, gpu_devices=gpu_devices)
        self.default_timeout = default_timeout

    def schedule(self, fn: Callable, args: Tuple, timeout: Optional[float] = None) -> TaskResult:
        if self.backend is None:
            raise RuntimeError("Please initialize parallel backend before calling schedule")
        if timeout is None:
            timeout = self.default_timeout
        return self.backend.schedule(fn, args, timeout)

    def close(self, force: bool = False):
        if self.backend is not None:
            self.backend.close(force)
            self.backend = None
