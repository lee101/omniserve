"""Tier-based VRAM arbitration across proxied upstreams.

Policy: paid API traffic (any modality) is highest, then subscribers, then
free, then background/batch which always yields. Because tier is per-REQUEST,
a paid image call and a batch image-gen script hit the same upstream at
different priorities. This exercises that a higher-tier request can push a
lower-tier VRAM-holding upstream out, but not the reverse.
"""
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


class VramProxy(Backend):
    """Stand-in for a GPU-backed proxy: holds resident_gib, releases on unload."""
    supports_sleep = False

    def __init__(self, spec, gpu):
        super().__init__(spec)
        self.gpu = gpu
        self._gib = spec.extra["resident_gib"]
        self.evicts = 0

    def resident_gib(self):
        return self._gib if self.state in (State.READY, State.LOADING) else 0.0

    def load(self):
        self.gpu.used += self._gib
        self.state = State.READY

    def unload(self):
        if self.state in (State.READY, State.LOADING):
            self.gpu.used = max(0.0, self.gpu.used - self._gib)
        self.evicts += 1
        self.state = State.UNLOADED

    def infer(self, request):
        return {"model": self.spec.key}


@pytest.fixture
def sched():
    gpu = FakeGpu(total=24.0)
    for key, gib in [("image", 15.0), ("llm", 13.0)]:
        register(ModelSpec(key=key, family="x", repo_id=key, engine="fake",
                           resident_gib=gib, extra={"resident_gib": gib}))

    s = Scheduler(backend_factory=lambda spec: VramProxy(spec, gpu),
                  vram_free=gpu.free, vram_total=lambda: gpu.total,
                  headroom_gib=2.0, start_reaper=False, tier_protect_s=120)
    s._test_gpu = gpu
    return s


def test_paid_llm_evicts_background_image(sched):
    # a batch image-gen script (background) warms the image server
    sched.infer("image", {}, Tier.BACKGROUND)
    assert sched.backends["image"].state == State.READY
    # a paid LLM request arrives; image + llm can't coexist (15+13+2 > 32)
    sched.infer("llm", {}, Tier.PAID)
    assert sched.backends["llm"].state == State.READY
    assert sched.backends["image"].state == State.UNLOADED  # background yielded
    assert sched.backends["image"].evicts == 1


def test_paid_image_not_evicted_by_free_llm(sched):
    # a PAID image API request warms the image server
    sched.infer("image", {}, Tier.PAID)
    # a free LLM request cannot push paid image out — gets a clean retry error
    with pytest.raises(CapacityError):
        sched.infer("llm", {}, Tier.FREE)
    assert sched.backends["image"].state == State.READY  # paid protected


def test_sub_beats_free_but_not_paid(sched):
    sched.infer("image", {}, Tier.SUB)      # subscriber warms image
    # free LLM cannot evict a subscriber's recent use
    with pytest.raises(CapacityError):
        sched.infer("llm", {}, Tier.FREE)
    assert sched.backends["image"].state == State.READY
    # but a paid LLM can
    sched.infer("llm", {}, Tier.PAID)
    assert sched.backends["llm"].state == State.READY
    assert sched.backends["image"].state == State.UNLOADED


def test_background_always_yields(sched):
    sched.infer("llm", {}, Tier.FREE)       # a free text-gen request holds llm
    # a background batch image job cannot evict even a free request
    with pytest.raises(CapacityError):
        sched.infer("image", {}, Tier.BACKGROUND)
    assert sched.backends["llm"].state == State.READY


def test_protection_expires_lets_free_reclaim(sched):
    sched.infer("image", {}, Tier.PAID)
    sched.tier_protect_s = 0.0              # paid use is now "old"
    sched.infer("llm", {}, Tier.FREE)       # free can reclaim stale VRAM
    assert sched.backends["llm"].state == State.READY
    assert sched.backends["image"].state == State.UNLOADED
