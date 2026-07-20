"""Priority admission for GPU work.

Four tiers, best first: PAID, SUB (unlimited subscription), FREE, BACKGROUND.
The gate hands out GPU slots strictly by tier (FIFO within a tier). BACKGROUND
is only admitted when nothing else is running or waiting — it is the
"while nothing else is running" class and is the first to starve under load.

Each waiter parks on its own Event and only the queue head is woken when state
changes, so a release wakes exactly one thread regardless of queue depth
(a shared-Condition broadcast is O(waiters) wakeups per event, O(n^2) overall).
"""
from __future__ import annotations

import enum
import heapq
import itertools
import threading
import time


class Tier(enum.IntEnum):
    PAID = 0
    SUB = 1
    FREE = 2
    BACKGROUND = 3

    @classmethod
    def parse(cls, value: str | None, default: "Tier" = None) -> "Tier":
        if default is None:
            default = cls.FREE
        if not value:
            return default
        try:
            return cls[value.strip().upper()]
        except KeyError:
            return default


class AdmissionTimeout(RuntimeError):
    pass


class PriorityGate:
    """Counting semaphore whose waiters are served in tier order.

    slots: concurrent GPU requests admitted. A streamed request keeps its slot
    until the client disconnects or the upstream finishes, so the scheduler
    cannot start a conflicting diffusion/model swap behind an active stream.
    """

    def __init__(self, slots: int = 1):
        self._slots = slots
        self._active = 0
        self._active_by_tier: dict[Tier, int] = {t: 0 for t in Tier}
        self._served = {t: 0 for t in Tier}
        self._waiting: list[tuple[int, int, threading.Event]] = []  # (tier, seq, event)
        self._seq = itertools.count()
        self._lock = threading.Lock()

    def _head_admissible(self) -> bool:
        """Caller holds the lock; may the current queue head run now?"""
        if not self._waiting or self._active >= self._slots:
            return False
        tier = self._waiting[0][0]
        if tier == int(Tier.BACKGROUND):
            # background runs only on an otherwise idle GPU; a background head
            # implies no better-tier waiters (heap order), so idle == no active
            return self._active == 0
        return True

    def _wake_head(self) -> None:
        if self._head_admissible():
            self._waiting[0][2].set()

    def acquire(self, tier: Tier, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            if not self._waiting and self._active < self._slots and (
                tier != Tier.BACKGROUND or self._active == 0
            ):
                self._admit(tier)
                return
            ev = threading.Event()
            entry = (int(tier), next(self._seq), ev)
            heapq.heappush(self._waiting, entry)
            # a new head (e.g. paid arriving behind an unadmissible background
            # head) may be runnable immediately
            self._wake_head()
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                with self._lock:
                    if entry in self._waiting:
                        self._waiting.remove(entry)
                        heapq.heapify(self._waiting)
                        self._wake_head()
                        raise AdmissionTimeout(
                            f"{tier.name} request timed out waiting for a GPU slot")
                    # we were popped right at the deadline: the slot is ours
                    return
            if not ev.wait(remaining):
                continue  # deadline check on next loop
            with self._lock:
                if self._waiting and self._waiting[0] is entry and self._head_admissible():
                    heapq.heappop(self._waiting)
                    self._admit(tier)
                    self._wake_head()  # more slots may remain for the next head
                    return
                if entry not in self._waiting:
                    return  # already admitted by a racing release
                ev.clear()  # spurious: state changed before we got the lock

    def _admit(self, tier: Tier) -> None:
        self._active += 1
        self._active_by_tier[tier] += 1
        self._served[tier] += 1

    def release(self, tier: Tier) -> None:
        with self._lock:
            self._active -= 1
            self._active_by_tier[tier] -= 1
            self._wake_head()

    def slot(self, tier: Tier, timeout: float | None = None):
        gate = self

        class _Slot:
            def __enter__(self):
                gate.acquire(tier, timeout)
                return self

            def __exit__(self, *exc):
                gate.release(tier)
                return False

        return _Slot()

    def stats(self) -> dict:
        with self._lock:
            waiting = {t.name.lower(): 0 for t in Tier}
            for t, _, _ in self._waiting:
                waiting[Tier(t).name.lower()] += 1
            return {
                "slots": self._slots,
                "active": self._active,
                "active_by_tier": {t.name.lower(): n for t, n in self._active_by_tier.items() if n},
                "waiting": waiting,
                "served": {t.name.lower(): n for t, n in self._served.items()},
            }
