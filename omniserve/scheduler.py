from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

from .backends.base import Backend, State, make_backend
from .catalog import ModelSpec, get_model
from .gpu import free_vram_gib, total_vram_gib

log = logging.getLogger("omniserve.scheduler")


class CapacityError(RuntimeError):
    pass


class Scheduler:
    def __init__(
        self,
        backend_factory: Callable[[ModelSpec], Backend] = make_backend,
        vram_free: Callable[[], float] = free_vram_gib,
        vram_total: Callable[[], float] = total_vram_gib,
        headroom_gib: float = float(os.environ.get("OMNISERVE_HEADROOM_GIB", "2")),
        idle_sleep_s: float = float(os.environ.get("OMNISERVE_IDLE_SLEEP_S", "300")),
        idle_unload_s: float = float(os.environ.get("OMNISERVE_IDLE_UNLOAD_S", "3600")),
        reaper_interval_s: float = 30.0,
        start_reaper: bool = True,
    ):
        self.backend_factory = backend_factory
        self.vram_free = vram_free
        self.vram_total = vram_total
        self.headroom_gib = headroom_gib
        self.idle_sleep_s = idle_sleep_s
        self.idle_unload_s = idle_unload_s
        self.backends: dict[str, Backend] = {}
        self.swap_lock = threading.RLock()
        self._stop = threading.Event()
        self._reaper = None
        if start_reaper:
            self._reaper = threading.Thread(target=self._reap_loop, args=(reaper_interval_s,), daemon=True)
            self._reaper.start()

    def _get(self, key: str) -> Backend:
        if key not in self.backends:
            with self.swap_lock:
                if key not in self.backends:
                    self.backends[key] = self.backend_factory(get_model(key))
        return self.backends[key]

    def _resident(self) -> list[Backend]:
        return [b for b in self.backends.values() if b.state in (State.READY, State.SLEEPING)]

    def _evict_for(self, needed_gib: float, protect: str) -> None:
        candidates = sorted(
            (b for b in self._resident() if b.spec.key != protect),
            key=lambda b: b.last_used,
        )
        for b in candidates:
            if self.vram_free() >= needed_gib:
                return
            log.info("evicting %s (state=%s, idle=%.0fs)", b.spec.key, b.state.value, time.time() - b.last_used)
            with b.lock:
                b.unload()
                b.state = State.UNLOADED
        if self.vram_free() < needed_gib:
            total = self.vram_total()
            if total and needed_gib > total:
                raise CapacityError(
                    f"model needs {needed_gib:.1f} GiB but GPU has {total:.1f} GiB total")

    def ensure(self, key: str) -> Backend:
        b = self._get(key)
        b.touch()
        if b.state == State.READY:
            return b
        with self.swap_lock:
            b.touch()
            if b.state == State.READY:
                return b
            if b.state == State.SLEEPING:
                log.info("waking %s", key)
                with b.lock:
                    b.wake()
                    b.state = State.READY
                return b
            needed = b.spec.resident_gib + self.headroom_gib
            self._evict_for(needed, protect=key)
            log.info("loading %s (%.1f GiB, free %.1f GiB)", key, b.spec.resident_gib, self.vram_free())
            b.state = State.LOADING
            try:
                with b.lock:
                    b.load()
                b.state = State.READY
            except Exception:
                b.state = State.UNLOADED
                try:
                    b.unload()
                except Exception:
                    pass
                raise
            return b

    def infer(self, key: str, request: dict) -> dict:
        b = self.ensure(key)
        try:
            with b.lock:
                b.touch()
                result = b.infer(request)
            b.touch()
            return result
        except Exception:
            if _is_oom(request, b):
                log.warning("oom on %s, evicting others and retrying", key)
                with self.swap_lock:
                    self._evict_for(b.spec.resident_gib + self.headroom_gib, protect=key)
                with b.lock:
                    return b.infer(request)
            raise

    def sleep(self, key: str) -> None:
        b = self.backends.get(key)
        if b and b.state == State.READY:
            with self.swap_lock, b.lock:
                if b.supports_sleep:
                    b.sleep()
                    b.state = State.SLEEPING
                else:
                    b.unload()
                    b.state = State.UNLOADED

    def stop(self, key: str) -> None:
        b = self.backends.get(key)
        if b and b.state != State.UNLOADED:
            with self.swap_lock, b.lock:
                b.unload()
                b.state = State.UNLOADED

    def status(self) -> dict:
        return {
            "vram_free_gib": round(self.vram_free(), 2),
            "vram_total_gib": round(self.vram_total(), 2),
            "backends": [b.info() for b in self.backends.values()],
        }

    def shutdown(self) -> None:
        self._stop.set()
        for key in list(self.backends):
            try:
                self.stop(key)
            except Exception:
                pass

    def _reap_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            now = time.time()
            for b in list(self.backends.values()):
                idle = now - b.last_used
                try:
                    if b.state == State.READY and idle > self.idle_sleep_s:
                        log.info("idle-sleep %s after %.0fs", b.spec.key, idle)
                        self.sleep(b.spec.key)
                    elif b.state == State.SLEEPING and idle > self.idle_unload_s:
                        log.info("idle-unload %s after %.0fs", b.spec.key, idle)
                        self.stop(b.spec.key)
                except Exception:
                    log.exception("reaper failed for %s", b.spec.key)


def _is_oom(request: dict, backend: Backend) -> bool:
    import sys
    exc = sys.exc_info()[1]
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text
