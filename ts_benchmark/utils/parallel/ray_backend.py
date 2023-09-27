# -*- coding: utf-8 -*-
from __future__ import absolute_import

import random
import sys
import itertools
import logging
import os
import queue
import threading
import time
from typing import Callable, Tuple, Any, List, NoReturn, Optional, Dict

import ray
from ray.exceptions import RayActorError

from ts_benchmark.utils.parallel.base import TaskResult

logger = logging.getLogger(__name__)

class RayActor:
    def __init__(self):
        self._idle = True
        self._start_time = None
        sys.path.insert(0,"ts_benchmark/baselines/third_party")

    def run(self, fn: Callable, args: Tuple) -> Any:
        self._start_time = time.time()
        self._idle = False
        res = fn(*args)
        self._idle = True
        return res

    def start_time(self) -> Optional[float]:
        return None if self._idle or self._start_time is None else self._start_time


class RayResult(TaskResult):
    __slots__ = ["_event", "_result"]

    def __init__(self, event: threading.Event):
        self._event = event
        self._result = None

    def put(self, value: Any) -> NoReturn:
        self._result = value
        self._event.set()

    def result(self) -> Any:
        self._event.wait()
        if isinstance(self._result, Exception):
            raise self._result
        else:
            return self._result


class RayTask:
    __slots__ = ["result", "actor_id", "timeout", "start_time"]

    def __init__(self, result: Any = None, actor_id: Optional[int] = None, timeout: float = -1):
        self.result = result
        self.actor_id = actor_id
        self.timeout = timeout
        self.start_time = None


class RayActorPool:
    """
    ray actor 资源池

    和 ray 的内置 ActorPool 不同，本实现试图支持为每个任务限时
    """

    def __init__(self, n_workers: int, per_worker_resources: Optional[Dict] = None):
        if per_worker_resources:
            per_worker_resources = {}

        self.actor_class = ray.remote(
            max_restarts=-1,
            num_cpus=per_worker_resources.get("num_cpus", 1),
            num_gpus=per_worker_resources.get("num_gpus", 0),
        )(RayActor)
        self.actors = [self.actor_class.options(max_concurrency=2).remote() for _ in range(n_workers)]

        # these data are only accessed in the main thread
        self._task_counter = itertools.count()

        # these data are only accessed in the loop thread
        self._task_info = {}
        self._ray_task_to_id = {}
        self._active_tasks = []
        self._idle_actors = list(range(len(self.actors)))
        self._restarting_actor_pool = {}

        # message path between threads
        self._is_closed = False
        self._idle_event = threading.Event()
        self._pending_queue = queue.Queue(maxsize=1000000)

        self._main_loop_thread = threading.Thread(target=self._main_loop)
        self._main_loop_thread.start()

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        self._idle_event.clear()
        task_id = next(self._task_counter)
        result = RayResult(threading.Event())
        self._pending_queue.put((fn, args, timeout, task_id, result), block=True)
        return result

    def _handle_ready_tasks(self, tasks: List) -> NoReturn:
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            try:
                task_info.result.put(ray.get(task_obj))
            except RayActorError as e:
                logger.info("task %d died unexpectedly on actor %d: %s", task_id, task_info.actor_id, e)
                task_info.result.put(RuntimeError(f"task died unexpectedly: {e}"))
            self._idle_actors.append(task_info.actor_id)
            del self._task_info[task_id]
            del self._ray_task_to_id[task_obj]

    def _get_duration(self, task_info: RayTask) -> Optional[float]:
        if task_info.start_time is None:
            try:
                task_info.start_time = ray.get(self.actors[task_info.actor_id].start_time.remote())
            except RayActorError as e:
                logger.info("actor %d died unexpectedly: %s, restarting...", task_info.actor_id, e)
                return None

        return -1 if task_info.start_time is None else time.time() - task_info.start_time
    
    def _handle_unfinished_tasks(self, tasks: List) -> NoReturn:
        new_active_tasks = []
        for task_obj in tasks:
            task_id = self._ray_task_to_id[task_obj]
            task_info = self._task_info[task_id]
            duration = self._get_duration(task_info)
            if duration is None or 0 < task_info.timeout < duration:
                if duration is not None:
                    ray.kill(self.actors[task_info.actor_id], no_restart=False)
                    logger.info("actor %d killed after timeout %f", task_info.actor_id, task_info.timeout)
                self._restarting_actor_pool[task_info.actor_id] = time.time()
                task_info.result.put(TimeoutError(f"time limit exceeded: {task_info.timeout}"))
                del self._task_info[task_id]
                del self._ray_task_to_id[task_obj]
            else:
                new_active_tasks.append(task_obj)
        self._active_tasks = new_active_tasks

    def _check_restarting_actors(self):
        new_restarting_pool = {}
        for actor_id, restart_time in self._restarting_actor_pool.items():
            if time.time() - restart_time > 5:
                self._idle_actors.append(actor_id)
            else:
                new_restarting_pool[actor_id] = restart_time
        self._restarting_actor_pool = new_restarting_pool

    def _main_loop(self) -> NoReturn:
        while not self._is_closed:
            self._check_restarting_actors()

            logger.debug(
                "%d active tasks, %d idle actors, %d restarting actors",
                len(self._active_tasks),
                len(self._idle_actors),
                len(self._restarting_actor_pool),
            )

            if not self._active_tasks and self._pending_queue.empty():
                self._idle_event.set()
                time.sleep(1)
                continue

            if self._active_tasks:
                ready_tasks, unfinished_tasks = ray.wait(self._active_tasks, timeout=1)
                self._handle_ready_tasks(ready_tasks)
                self._handle_unfinished_tasks(unfinished_tasks)

            while self._idle_actors and not self._pending_queue.empty():
                fn, args, timeout, task_id, result = self._pending_queue.get_nowait()
                cur_actor = self._idle_actors.pop()
                task_obj = self.actors[cur_actor].run.remote(fn, args)
                self._task_info[task_id] = RayTask(result=result, actor_id=cur_actor, timeout=timeout)
                self._ray_task_to_id[task_obj] = task_id
                self._active_tasks.append(task_obj)
                logger.debug("task %d assigned to actor %d", task_id, cur_actor)

    def wait(self) -> NoReturn:
        if self._is_closed:
            return
        if self._pending_queue.empty() and not self._active_tasks:
            return
        self._idle_event.clear()
        self._idle_event.wait()

    def close(self) -> NoReturn:
        self._is_closed = True
        for actor in self.actors:
            ray.kill(actor)
        self._main_loop_thread.join()


class RayBackend:

    def __init__(self, n_workers: Optional[int] = None, n_cpus: Optional[int] = None, gpu_devices: Optional[List[int]] = None):
        if n_cpus is None:
            n_cpus = os.cpu_count()
        if n_workers is None:
            n_workers = n_cpus
        if gpu_devices is None:
            gpu_devices = []

        cpu_per_worker = n_cpus / n_workers
        gpu_per_worker = len(gpu_devices) / n_workers

        if not ray.is_initialized():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
            ray.init(num_cpus=n_cpus, num_gpus=len(gpu_devices))

        self.pool = RayActorPool(
            n_workers,
            {
                "num_cpus": cpu_per_worker,
                "num_gpus": gpu_per_worker,
            }
        )

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> RayResult:
        return self.pool.schedule(fn, args, timeout)

    def close(self, force: bool = False):
        if not force:
            self.pool.wait()
        self.pool.close()
        ray.shutdown()


if __name__ == "__main__":
    backend = RayBackend(3)


    def sleep_func(t):
        time.sleep(t)
        print(f"sleep after {t}")
        return t

    results = []
    results.append(backend.schedule(sleep_func, (10,), timeout=5))
    results.append(backend.schedule(sleep_func, (10,), timeout=20))
    results.append(backend.schedule(sleep_func, (1,), timeout=5))
    results.append(backend.schedule(sleep_func, (2,), timeout=5))
    results.append(backend.schedule(sleep_func, (3,), timeout=5))
    results.append(backend.schedule(sleep_func, (4,), timeout=5))
    results.append(backend.schedule(sleep_func, (5,), timeout=5))
    results.append(backend.schedule(sleep_func, (6,), timeout=5))

    for i, res in enumerate(results):
        try:
            print(f"{i}-th task result: {res.result()}")
        except TimeoutError:
            print(f"{i}-th task fails after timeout")

    backend.close()
    # time.sleep(100)
    # time.sleep(1)
    # pool.wait()
    # pool.close()
