from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GpuStat:
    index: int
    total_gib: float
    free_gib: float
    name: str = ""


def query_gpus() -> list[GpuStat]:
    try:
        import torch
        if torch.cuda.is_available():
            stats = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                stats.append(GpuStat(i, total / 2**30, free / 2**30, torch.cuda.get_device_name(i)))
            return stats
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.total,memory.free,name",
                 "--format=csv,noheader,nounits"], text=True, timeout=10)
            stats = []
            for line in out.strip().splitlines():
                idx, total, free, name = [p.strip() for p in line.split(",", 3)]
                stats.append(GpuStat(int(idx), float(total) / 1024, float(free) / 1024, name))
            return stats
        except Exception:
            pass
    return []


def free_vram_gib(device: int = 0) -> float:
    for g in query_gpus():
        if g.index == device:
            return g.free_gib
    return 0.0


def total_vram_gib(device: int = 0) -> float:
    for g in query_gpus():
        if g.index == device:
            return g.total_gib
    return 0.0


def free_disk_gib(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 2**30
