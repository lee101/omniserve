from __future__ import annotations

import threading
import time
from enum import Enum

from ..catalog import ModelSpec


class State(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    SLEEPING = "sleeping"


class Backend:
    supports_sleep = False
    # HTTP subprocess/proxy backends can safely accept concurrent requests.
    # In-process torch pipelines normally cannot, so they keep the default.
    concurrent_requests = False

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.state = State.UNLOADED
        self.last_used = 0.0
        self.lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self._active_requests = 0

    def touch(self) -> None:
        self.last_used = time.time()

    def begin_request(self) -> None:
        with self._activity_lock:
            self._active_requests += 1
            self.touch()

    def end_request(self) -> None:
        with self._activity_lock:
            if self._active_requests <= 0:
                raise RuntimeError(f"unbalanced request accounting for {self.spec.key}")
            self._active_requests -= 1
            self.touch()

    def active_requests(self) -> int:
        with self._activity_lock:
            return self._active_requests

    def resident_gib(self) -> float:
        return self.spec.resident_gib if self.state in (State.READY, State.LOADING) else 0.0

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def sleep(self) -> None:
        self.unload()

    def wake(self) -> None:
        self.load()

    def infer(self, request: dict) -> dict:
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "model": self.spec.key,
            "family": self.spec.family,
            "engine": self.spec.engine,
            "state": self.state.value,
            "last_used": self.last_used,
            "resident_gib": self.resident_gib(),
            "active_requests": self.active_requests(),
        }


ENGINES: dict[str, type[Backend]] = {}


def register_engine(name: str):
    def deco(cls: type[Backend]) -> type[Backend]:
        ENGINES[name] = cls
        return cls
    return deco


def make_backend(spec: ModelSpec) -> Backend:
    if spec.engine not in ENGINES:
        raise KeyError(f"no engine '{spec.engine}' (available: {sorted(ENGINES)})")
    return ENGINES[spec.engine](spec)
