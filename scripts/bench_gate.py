#!/usr/bin/env python3
"""PriorityGate contention benchmark: throughput, wait latency, memory.

Simulates the shared-GPU admission pattern: many threads across mixed tiers
grabbing 1-2 slots with a tiny hold time. Reports acquisitions/sec, p50/p99
wait, RSS delta, and tracemalloc peak. Run before/after scheduler changes:

    python scripts/bench_gate.py --threads 64 --iters 200 --slots 2
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time
import tracemalloc

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omniserve.priority import AdmissionTimeout, PriorityGate, Tier  # noqa: E402

TIER_MIX = [Tier.PAID, Tier.SUB, Tier.FREE, Tier.FREE, Tier.FREE, Tier.SUB]


def rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--hold-us", type=int, default=50)
    args = ap.parse_args()

    gate = PriorityGate(slots=args.slots)
    waits: list[list[float]] = [[] for _ in range(args.threads)]
    errors = [0] * args.threads
    hold_s = args.hold_us / 1e6
    start_evt = threading.Event()

    def worker(idx: int):
        tier = TIER_MIX[idx % len(TIER_MIX)]
        my_waits = waits[idx]
        start_evt.wait()
        for _ in range(args.iters):
            t0 = time.perf_counter()
            try:
                gate.acquire(tier, timeout=60)
            except AdmissionTimeout:
                errors[idx] += 1
                continue
            my_waits.append(time.perf_counter() - t0)
            time.sleep(hold_s)
            gate.release(tier)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
    for t in threads:
        t.start()

    rss0 = rss_mib()
    tracemalloc.start()
    wall0 = time.perf_counter()
    start_evt.set()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_waits = sorted(w for ws in waits for w in ws)
    n = len(all_waits)
    print(f"threads={args.threads} iters={args.iters} slots={args.slots} hold={args.hold_us}us")
    print(f"acquisitions: {n} in {wall:.2f}s -> {n / wall:,.0f}/s (timeouts: {sum(errors)})")
    if n:
        print(f"wait p50={all_waits[n // 2] * 1e3:.2f}ms "
              f"p99={all_waits[int(n * 0.99)] * 1e3:.2f}ms "
              f"max={all_waits[-1] * 1e3:.1f}ms "
              f"mean={statistics.mean(all_waits) * 1e3:.2f}ms")
    print(f"rss now {rss_mib():.1f} MiB (was {rss0:.1f}), tracemalloc peak {tm_peak / 1e6:.1f} MB")
    print(f"stats: {gate.stats()}")


if __name__ == "__main__":
    main()
