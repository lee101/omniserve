from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MIRROR = os.environ.get("OMNISERVE_MODELS_BASE", "https://appstatic.app.nz/models")


def weights_dir() -> Path:
    for cand in (os.environ.get("WEIGHTS_DIR"), "/runpod-volume/models", "/weights"):
        if cand and Path(cand).is_dir():
            return Path(cand)
    d = Path(os.environ.get("OMNISERVE_HOME", Path.home() / ".cache/omniserve")) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=120) as r, open(part, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if expected_size is not None and part.stat().st_size != expected_size:
        part.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {url}")
    os.replace(part, dest)


def _mirror_fetch(repo_id: str, dest: Path, mirror: str) -> bool:
    manifest = _fetch_json(f"{mirror}/{repo_id}/manifest.json")
    if not manifest:
        return False
    files = manifest.get("files", [])
    if not files:
        return False

    def one(item: dict) -> None:
        rel = item["path"] if isinstance(item, dict) else item
        size = item.get("size") if isinstance(item, dict) else None
        target = dest / rel
        if target.exists() and (size is None or target.stat().st_size == size):
            return
        _download_file(f"{mirror}/{repo_id}/{rel}", target, size)

    with ThreadPoolExecutor(max_workers=int(os.environ.get("OMNISERVE_DL_WORKERS", "12"))) as ex:
        list(ex.map(one, files))
    return True


def resolve_weights(repo_id: str, single_file: str = "", allow_download: bool = True) -> Path:
    base = weights_dir()
    local = base / repo_id
    marker = local / ".incomplete"
    if single_file:
        f = local / single_file
        if f.exists() and not marker.exists():
            return f
    elif local.is_dir() and any(local.iterdir()) and not marker.exists():
        return local

    if not allow_download:
        raise FileNotFoundError(f"{repo_id} not cached under {base}")

    local.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        mirror = os.environ.get("OMNISERVE_MODELS_BASE", DEFAULT_MIRROR)
        if mirror and _mirror_fetch(repo_id, local, mirror):
            marker.unlink(missing_ok=True)
            return local / single_file if single_file else local

        from huggingface_hub import hf_hub_download, snapshot_download
        token = os.environ.get("HF_TOKEN")
        if single_file:
            hf_hub_download(repo_id, single_file, local_dir=local, token=token)
        else:
            snapshot_download(repo_id, local_dir=local, token=token)
        marker.unlink(missing_ok=True)
        return local / single_file if single_file else local
    except Exception:
        raise
