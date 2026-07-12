import threading
import time

import pytest

from omniserve.backends.base import Backend, State
from omniserve.catalog import ModelSpec, register
from omniserve.scheduler import CapacityError, Scheduler

TOTAL = 24.0


class FakeGpu:
    def __init__(self, total=TOTAL):
        self.total = total
        self.used = 0.0

    def free(self):
        return self.total - self.used


class FakeBackend(Backend):
    supports_sleep = True

    def __init__(self, spec, gpu):
        super().__init__(spec)
        self.gpu = gpu
        self.loads = 0
        self.sleeps = 0
        self.wakes = 0

    def load(self):
        self.gpu.used += self.spec.resident_gib
        self.loads += 1

    def unload(self):
        if self.state in (State.READY, State.SLEEPING, State.LOADING):
            self.gpu.used -= self.spec.resident_gib
            self.gpu.used = max(0.0, self.gpu.used)

    def sleep(self):
        self.gpu.used -= self.spec.resident_gib
        self.sleeps += 1

    def wake(self):
        self.gpu.used += self.spec.resident_gib
        self.wakes += 1

    def infer(self, request):
        return {"echo": request, "model": self.spec.key}


@pytest.fixture
def sched():
    gpu = FakeGpu()
    for key, gib in [("small-a", 8.0), ("small-b", 8.0), ("big", 20.0)]:
        register(ModelSpec(key=key, family="test", repo_id=f"t/{key}", engine="fake", resident_gib=gib))
    register(ModelSpec(key="too-big", family="test", repo_id="t/too-big", engine="fake", resident_gib=100.0))
    backends = {}

    def factory(spec):
        b = FakeBackend(spec, gpu)
        backends[spec.key] = b
        return b

    s = Scheduler(backend_factory=factory, vram_free=gpu.free, vram_total=lambda: gpu.total,
                  headroom_gib=2.0, start_reaper=False)
    s._test_gpu = gpu
    s._test_backends = backends
    return s


def test_load_and_infer(sched):
    out = sched.infer("small-a", {"x": 1})
    assert out["model"] == "small-a"
    assert sched.backends["small-a"].state == State.READY


def test_second_model_coexists(sched):
    sched.ensure("small-a")
    sched.ensure("small-b")
    assert sched.backends["small-a"].state == State.READY
    assert sched.backends["small-b"].state == State.READY


def test_eviction_lru(sched):
    sched.ensure("small-a")
    time.sleep(0.01)
    sched.ensure("small-b")
    time.sleep(0.01)
    sched.backends["small-b"].touch()
    sched.ensure("big")
    assert sched.backends["big"].state == State.READY
    assert sched.backends["small-a"].state == State.UNLOADED
    assert sched.backends["small-b"].state == State.UNLOADED


def test_capacity_error(sched):
    with pytest.raises(CapacityError):
        sched.ensure("too-big")


def test_sleep_wake(sched):
    sched.ensure("small-a")
    sched.sleep("small-a")
    assert sched.backends["small-a"].state == State.SLEEPING
    assert sched._test_gpu.used == 0.0
    sched.infer("small-a", {})
    assert sched.backends["small-a"].state == State.READY
    assert sched.backends["small-a"].wakes == 1
    assert sched.backends["small-a"].loads == 1


def test_ensure_idempotent(sched):
    sched.ensure("small-a")
    sched.ensure("small-a")
    assert sched.backends["small-a"].loads == 1


def test_concurrent_ensure_single_load(sched):
    errs = []

    def hit():
        try:
            sched.infer("small-a", {})
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errs
    assert sched.backends["small-a"].loads == 1


def test_reaper_tiers():
    gpu = FakeGpu()
    register(ModelSpec(key="reap-me", family="test", repo_id="t/reap", engine="fake", resident_gib=4.0))
    s = Scheduler(backend_factory=lambda spec: FakeBackend(spec, gpu), vram_free=gpu.free,
                  vram_total=lambda: gpu.total, idle_sleep_s=0.05, idle_unload_s=0.15,
                  reaper_interval_s=0.03, start_reaper=True)
    s.ensure("reap-me")
    time.sleep(0.12)
    assert s.backends["reap-me"].state == State.SLEEPING
    time.sleep(0.25)
    assert s.backends["reap-me"].state == State.UNLOADED
    s.shutdown()
