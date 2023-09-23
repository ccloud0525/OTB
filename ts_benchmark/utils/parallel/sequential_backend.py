# -*- coding: utf-8 -*-

from __future__ import absolute_import

import warnings
from typing import Tuple, Any, NoReturn, Callable

from ts_benchmark.utils.parallel.base import TaskResult


class SequentialResult(TaskResult):

    def __init__(self):
        self._result = None

    def result(self) -> Any:
        return self._result

    def put(self, value: Any) -> NoReturn:
        self._result = value


class SequentialBackend:

    def __init__(self, *args, **kwargs):
        super().__init__()

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> SequentialResult:
        if timeout != -1:
            warnings.warn("timeout is not supported by SequentialBackend, ignoring")
        res = SequentialResult()
        res.put(fn(*args))
        return res

    def close(self, force: bool = False):
        pass
