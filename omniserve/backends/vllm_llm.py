from __future__ import annotations

import logging
import os
import shlex
import signal
import socket
import subprocess
import time

import httpx

from .base import Backend, register_engine

log = logging.getLogger("omniserve.vllm")


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


@register_engine("vllm")
class VllmBackend(Backend):
    supports_sleep = True

    def __init__(self, spec):
        super().__init__(spec)
        self.port = int(spec.extra.get("port", os.environ.get("OMNISERVE_VLLM_PORT", "8710")))
        self.proc: subprocess.Popen | None = None
        self._client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=600)

    def _cmd(self) -> list[str]:
        custom = self.spec.extra.get("cmd")
        if custom:
            return shlex.split(custom)
        args = [
            "vllm", "serve", self.spec.repo_id,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.spec.extra.get("max_model_len", 8192)),
            "--gpu-memory-utilization", str(self.spec.extra.get("gpu_mem_util", 0.85)),
            "--enable-prefix-caching",
            "--enable-sleep-mode",
        ]
        quant = os.environ.get("OMNISERVE_LLM_QUANT", self.spec.recommended_quant)
        if quant:
            args += ["--quantization", quant]
        return args

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited rc={self.proc.returncode}")
            try:
                r = self._client.get("/v1/models", timeout=5)
                if r.status_code == 200:
                    served = [m["id"] for m in r.json().get("data", [])]
                    if any(self.spec.repo_id in s or s in self.spec.repo_id for s in served) or served:
                        return
            except Exception:
                pass
            time.sleep(1.5)
        raise TimeoutError(f"vllm for {self.spec.key} not ready in {timeout}s")

    def load(self) -> None:
        if not _port_free(self.port):
            self._kill_port_orphans()
        env = dict(os.environ)
        env.setdefault("VLLM_SERVER_DEV_MODE", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")
        cmd = self._cmd()
        log.info("starting vllm: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        self._wait_ready(float(self.spec.extra.get("startup_timeout", 600)))

    def unload(self) -> None:
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait(timeout=30)
            self.proc = None

    def sleep(self) -> None:
        try:
            self._client.post("/sleep", params={"level": "1"}, timeout=60)
        except Exception:
            log.warning("sleep failed for %s, killing", self.spec.key)
            self.unload()

    def wake(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.load()
            return
        self._client.post("/wake_up", timeout=120)

    def resident_gib(self) -> float:
        from .base import State
        return self.spec.resident_gib if self.state in (State.READY, State.LOADING) else 0.0

    def _kill_port_orphans(self) -> None:
        try:
            out = subprocess.check_output(["lsof", "-ti", f"tcp:{self.port}"], text=True, timeout=10)
            for pid in out.split():
                try:
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
                except Exception:
                    os.kill(int(pid), signal.SIGKILL)
            time.sleep(1)
        except Exception:
            pass

    def infer(self, request: dict) -> dict:
        path = request.get("_path", "/v1/chat/completions")
        payload = {k: v for k, v in request.items() if not k.startswith("_")}
        payload["model"] = self.spec.repo_id
        r = self._client.post(path, json=payload)
        r.raise_for_status()
        return r.json()

    def proxy_stream(self, path: str, payload: dict):
        payload = dict(payload)
        payload["model"] = self.spec.repo_id
        payload["stream"] = True
        with self._client.stream("POST", path, json=payload) as r:
            for line in r.iter_lines():
                if line:
                    yield line + "\n\n"
