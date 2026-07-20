"""Proxy engine — front an existing upstream server without re-hosting the model.

The pragmatic path to one shared front door on a GPU that cannot hold duplicate
copies of every model: omniserve owns priority scheduling + admission, and each
catalog entry with ``engine="proxy"`` forwards to a battle-tested upstream
(the current image server, the vLLM adapter, the TTS/STT service, …).

A proxy backend holds no VRAM of its own (``resident_gib`` reports 0), so it
never competes in the scheduler's eviction accounting — the upstream manages
its own residency. What omniserve adds on top is the tier gate: a paid request
still jumps ahead of free/background before the bytes ever reach the upstream.

Config via the catalog spec's ``extra``:
    {"engine": "proxy", "extra": {"base_url": "http://127.0.0.1:8300",
                                   "model_override": "gemma-roleplay-v2"}}
``base_url`` may also come from OMNISERVE_PROXY_<KEY> env (key upper-snaked).
"""
from __future__ import annotations

import os

import httpx

from ..catalog import ModelSpec
from .base import Backend, State, register_engine


def _base_url_for(spec: ModelSpec) -> str:
    env_key = "OMNISERVE_PROXY_" + spec.key.upper().replace("-", "_").replace("/", "_")
    if os.environ.get(env_key):
        return os.environ[env_key]
    extra = getattr(spec, "extra", None) or {}
    if extra.get("base_url"):
        return extra["base_url"]
    raise RuntimeError(f"proxy '{spec.key}' has no base_url (set {env_key} or spec.extra.base_url)")


@register_engine("proxy")
class ProxyBackend(Backend):
    supports_sleep = False  # nothing local to sleep; upstream owns residency
    concurrent_requests = True

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._base = _base_url_for(spec).rstrip("/")
        extra = getattr(spec, "extra", None) or {}
        self._model_override = extra.get("model_override")
        self._timeout = float(extra.get("timeout", 600))
        self._client = httpx.Client(base_url=self._base, timeout=self._timeout)
        # A GPU-backed upstream (image server, vLLM) can be arbitrated for VRAM:
        # declare how much it holds and how to make it release. Then this proxy
        # participates in the scheduler's tier-protected eviction, so a paid
        # request (any modality) can push a lower-tier upstream out of VRAM.
        # CPU/remote upstreams (TTS, Gemini STT) leave resident_gib at 0 and
        # never take part.
        self._resident = float(extra.get("resident_gib", 0.0))
        self._evict_url = extra.get("evict_url", "")   # e.g. http://127.0.0.1:8100/admin/unload
        self._evict_method = (extra.get("evict_method") or "POST").upper()
        # A GPU-backed upstream may hold VRAM from traffic that bypassed
        # omniserve (direct hits to its port). So assume it's warm until we
        # explicitly evict it — that makes it an eviction candidate immediately.
        # The scheduler confirms against real free VRAM, so a wasted evict on an
        # already-free upstream is harmless (idempotent POST, no state change).
        if self._evict_url and self._resident > 0:
            self.state = State.READY

    def resident_gib(self) -> float:
        # Counts whenever we haven't explicitly evicted it — covers out-of-band
        # residency. Eviction candidate selection uses this to know how much
        # freeing us could recover; the real vram_free() is the source of truth.
        return self._resident if self.state != State.UNLOADED else 0.0

    def load(self) -> None:
        # readiness = upstream reachable. Cheap HEAD/GET; tolerate 404 (up but
        # no such route) as "reachable".
        try:
            self._client.get("/health")
        except httpx.HTTPError:
            pass  # some upstreams have no /health; the first real call will surface errors
        self.state = State.READY

    def unload(self) -> None:
        # For a VRAM-holding upstream, "unload" means ask it to release VRAM
        # (the actual free is confirmed by the scheduler's real vram_free()).
        # For a plain proxy this is a no-op.
        if self._evict_url and self._resident > 0:
            try:
                self._client.request(self._evict_method, self._evict_url, timeout=60)
            except httpx.HTTPError:
                pass  # best-effort; scheduler re-checks real free VRAM after
        self.state = State.UNLOADED
        self.state = State.READY

    def unload(self) -> None:
        self.state = State.UNLOADED

    def _payload(self, request: dict) -> tuple[str, dict]:
        path = request.get("_path", "/v1/chat/completions")
        payload = {k: v for k, v in request.items() if not k.startswith("_")}
        if self._model_override:
            payload["model"] = self._model_override
        return path, payload

    def infer(self, request: dict) -> dict:
        path, payload = self._payload(request)
        headers = dict(request.get("_headers") or {})
        raw_body = request.get("_raw_body")
        if raw_body is not None:
            content_type = request.get("_content_type")
            if content_type:
                headers["content-type"] = content_type
            r = self._client.post(path, content=raw_body, headers=headers)
        else:
            r = self._client.post(path, json=payload, headers=headers)
        if r.status_code >= 400:
            # surface the upstream status (401/403/422/...) instead of a 500
            from fastapi import HTTPException
            detail: object = r.text[:500]
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            raise HTTPException(r.status_code, detail)
        ct = r.headers.get("content-type", "")
        if ct.startswith("application/json"):
            return r.json()
        # binary modality (audio/image bytes): wrap so the server can pass through
        return {"_raw": r.content, "_content_type": ct}

    def proxy_stream(self, path: str, payload: dict, headers: dict | None = None):
        payload = dict(payload)
        if self._model_override:
            payload["model"] = self._model_override
        payload["stream"] = True
        with self._client.stream("POST", path, json=payload, headers=headers or {}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield line + "\n\n"
