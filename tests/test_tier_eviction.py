import time

import pytest

from omniserve.backends.base import Backend, State
from omniserve.catalog import ModelSpec, register
from omniserve.priority import Tier
from omniserve.scheduler import CapacityError, Scheduler


class FakeGpu:
    def __init__(self, total=24.0):
        self.total = total
        self.used = 0.0

    def free(self):
        return self.total - self.used


class FakeBackend(Backend):
    supports_sleep = False

    def __init__(self, spec, gpu):
        super().__init__(spec)
        self.gpu = gpu

    def load(self):
        self.gpu.used += self.spec.resident_gib

    def unload(self):
        if self.state in (State.READY, State.SLEEPING, State.LOADING):
            self.gpu.used = max(0.0, self.gpu.used - self.spec.resident_gib)

    def infer(self, request):
        return {"model": self.spec.key}


@pytest.fixture
def sched():
    gpu = FakeGpu()
    for key in ("tier-paid-model", "tier-free-model", "tier-other-model"):
        register(ModelSpec(key=key, family="test", repo_id=f"t/{key}", engine="fake", resident_gib=10.0))

    s = Scheduler(
        backend_factory=lambda spec: FakeBackend(spec, gpu),
        vram_free=gpu.free, vram_total=lambda: gpu.total,
        headroom_gib=2.0, start_reaper=False, tier_protect_s=60.0,
    )
    s._test_gpu = gpu
    return s


def test_free_cannot_evict_recent_paid(sched):
    sched.infer("tier-paid-model", {}, Tier.PAID)
    sched.infer("tier-free-model", {}, Tier.FREE)
    # both resident (20 GiB); a third 10 GiB model needs an eviction. The FREE
    # request may evict the free model but must not touch the paid one.
    sched.infer("tier-other-model", {}, Tier.FREE)
    assert sched.backends["tier-paid-model"].state == State.READY
    assert sched.backends["tier-free-model"].state == State.UNLOADED


def test_free_blocked_when_only_paid_evictable(sched):
    sched.infer("tier-paid-model", {}, Tier.PAID)
    sched.infer("tier-free-model", {}, Tier.PAID)  # paid used both models
    with pytest.raises(CapacityError):
        sched.infer("tier-other-model", {}, Tier.FREE)


def test_paid_evicts_free(sched):
    sched.infer("tier-paid-model", {}, Tier.FREE)
    sched.infer("tier-free-model", {}, Tier.FREE)
    sched.infer("tier-other-model", {}, Tier.PAID)
    assert sched.backends["tier-other-model"].state == State.READY


def test_protection_expires(sched):
    sched.infer("tier-paid-model", {}, Tier.PAID)
    sched.infer("tier-free-model", {}, Tier.PAID)
    sched.tier_protect_s = 0.01
    time.sleep(0.05)
    sched.infer("tier-other-model", {}, Tier.FREE)  # protection window over
    assert sched.backends["tier-other-model"].state == State.READY


def test_status_reports_tiers_and_admission(sched):
    sched.infer("tier-paid-model", {}, Tier.PAID)
    st = sched.status()
    assert st["admission"]["served"]["paid"] == 1
    tiers = {b["model"]: b["tier"] for b in st["backends"]}
    assert tiers["tier-paid-model"] == "paid"
