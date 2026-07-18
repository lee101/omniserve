import os

import pytest

from omniserve.backends.base import State, make_backend
from omniserve.catalog import ModelSpec, register, register_proxy_defaults, MODEL_CATALOG


def _spec(base_url):
    return ModelSpec(key="proxy-test", family="llm", repo_id="proxy-test",
                     engine="proxy", resident_gib=0.0,
                     extra={"base_url": base_url, "model_override": "up-model"})


class _FakeResp:
    def __init__(self, json_body=None, content=b"", ct="application/json", status=200):
        self._json = json_body
        self.content = content
        self.headers = {"content-type": ct}
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


def test_proxy_holds_no_vram_and_readies():
    b = make_backend(_spec("http://127.0.0.1:59999"))
    assert b.resident_gib() == 0.0          # never competes for VRAM
    b.load()                                # unreachable upstream tolerated
    assert b.state == State.READY


def test_proxy_forwards_json_with_model_override(monkeypatch):
    b = make_backend(_spec("http://up"))
    seen = {}

    def fake_post(path, json):
        seen["path"] = path
        seen["json"] = json
        return _FakeResp(json_body={"ok": True, "model": json["model"]})

    monkeypatch.setattr(b._client, "post", fake_post)
    out = b.infer({"_path": "/v1/chat/completions", "messages": [], "model": "ignored"})
    assert out == {"ok": True, "model": "up-model"}
    assert seen["path"] == "/v1/chat/completions"
    assert seen["json"]["model"] == "up-model"        # override applied
    assert "_path" not in seen["json"]                # private keys stripped


def test_proxy_binary_passthrough(monkeypatch):
    b = make_backend(_spec("http://up"))
    monkeypatch.setattr(b._client, "post",
                        lambda path, json: _FakeResp(content=b"RIFFxxxx", ct="audio/wav"))
    out = b.infer({"_path": "/v1/audio/speech", "input": "hi"})
    assert out["_raw"] == b"RIFFxxxx"
    assert out["_content_type"] == "audio/wav"


def test_register_proxy_defaults_from_env(monkeypatch):
    monkeypatch.setenv("OMNISERVE_PROXY_PROXY_TTS", "http://127.0.0.1:9080")
    monkeypatch.setenv("OMNISERVE_PROXY_PROXY_TTS_MODEL", "supertonic")
    register_proxy_defaults()
    assert "proxy-tts" in MODEL_CATALOG
    assert MODEL_CATALOG["proxy-tts"].family == "tts"
    assert MODEL_CATALOG["proxy-tts"].extra["base_url"] == "http://127.0.0.1:9080"
