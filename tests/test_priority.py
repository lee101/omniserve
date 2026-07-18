import threading
import time

import pytest

from omniserve.priority import AdmissionTimeout, PriorityGate, Tier


def test_tier_parse():
    assert Tier.parse("paid") == Tier.PAID
    assert Tier.parse("SUB") == Tier.SUB
    assert Tier.parse("background") == Tier.BACKGROUND
    assert Tier.parse(None) == Tier.FREE
    assert Tier.parse("garbage") == Tier.FREE


def test_paid_admitted_before_earlier_free():
    gate = PriorityGate(slots=1)
    order = []
    gate.acquire(Tier.FREE)  # occupy the slot

    def worker(tier, name):
        gate.acquire(tier)
        order.append(name)
        time.sleep(0.02)
        gate.release(tier)

    free_t = threading.Thread(target=worker, args=(Tier.FREE, "free"))
    free_t.start()
    time.sleep(0.05)  # free is queued first
    paid_t = threading.Thread(target=worker, args=(Tier.PAID, "paid"))
    paid_t.start()
    time.sleep(0.05)
    gate.release(Tier.FREE)  # free the slot; paid should win despite arriving later
    paid_t.join(2)
    free_t.join(2)
    assert order == ["paid", "free"]


def test_background_only_when_idle():
    gate = PriorityGate(slots=1)
    gate.acquire(Tier.FREE)
    with pytest.raises(AdmissionTimeout):
        gate.acquire(Tier.BACKGROUND, timeout=0.1)
    gate.release(Tier.FREE)
    gate.acquire(Tier.BACKGROUND, timeout=1)  # idle now
    gate.release(Tier.BACKGROUND)


def test_background_yields_to_waiting_free():
    gate = PriorityGate(slots=2)
    gate.acquire(Tier.FREE)
    # one slot remains, but background must not take it while free work runs
    with pytest.raises(AdmissionTimeout):
        gate.acquire(Tier.BACKGROUND, timeout=0.1)
    gate.release(Tier.FREE)


def test_fifo_within_tier():
    gate = PriorityGate(slots=1)
    order = []
    gate.acquire(Tier.FREE)

    def worker(name):
        gate.acquire(Tier.FREE)
        order.append(name)
        gate.release(Tier.FREE)

    threads = []
    for name in ("a", "b", "c"):
        t = threading.Thread(target=worker, args=(name,))
        t.start()
        threads.append(t)
        time.sleep(0.03)
    gate.release(Tier.FREE)
    for t in threads:
        t.join(2)
    assert order == ["a", "b", "c"]


def test_timeout_removes_waiter():
    gate = PriorityGate(slots=1)
    gate.acquire(Tier.PAID)
    with pytest.raises(AdmissionTimeout):
        gate.acquire(Tier.FREE, timeout=0.05)
    gate.release(Tier.PAID)
    gate.acquire(Tier.FREE, timeout=1)  # queue must be clean after the timeout
    gate.release(Tier.FREE)
    stats = gate.stats()
    assert stats["active"] == 0
    assert all(v == 0 for v in stats["waiting"].values())
