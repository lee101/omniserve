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

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._base = _base_url_for(spec).rstrip("/")
        extra = getattr(spec, "extra", None) or {}
        self._model_override = extra.get("model_override")
        self._timeout = float(extra.get("timeout", 600))
        self._client = httpx.Client(base_url=self._base, timeout=self._timeout)

    def resident_gib(self) -> float:
        return 0.0  # upstream holds the weights, not us

    def load(self) -> None:
        # readiness = upstream reachable. Cheap HEAD/GET; tolerate 404 (up but
        # no such route) as "reachable".
        try:
            self._client.get("/health")
        except httpx.HTTPError:
            pass  # some upstreams have no /health; the first real call will surface errors
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
        headers = request.get("_headers") or {}
        r = self._client.post(path, json=payload, headers=headers)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if ct.startswith("application/json"):
            return r.json()
        # binary modality (audio/image bytes): wrap so the server can pass through
        return {"_raw": r.content, "_content_type": ct}

    def proxy_stream(self, path: str, payload: dict):
        payload = dict(payload)
        if self._model_override:
            payload["model"] = self._model_override
        payload["stream"] = True
        with self._client.stream("POST", path, json=payload) as r:
            for line in r.iter_lines():
                if line:
                    yield line + "\n\n"
