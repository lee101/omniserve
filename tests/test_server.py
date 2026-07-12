import base64

import pytest
from fastapi.testclient import TestClient

from omniserve.backends.base import Backend, register_engine
from omniserve.catalog import ModelSpec, register
from omniserve.scheduler import Scheduler
from omniserve.server import create_app


@register_engine("stub-llm")
class StubLlm(Backend):
    def load(self):
        pass

    def unload(self):
        pass

    def infer(self, request):
        msg = request["messages"][-1]["content"]
        return {"id": "cmpl-1", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"echo:{msg}"},
                             "finish_reason": "stop"}]}


@register_engine("stub-image")
class StubImage(Backend):
    def load(self):
        pass

    def unload(self):
        pass

    def infer(self, request):
        n = request.get("n", 1)
        return {"images_b64": [base64.b64encode(b"fakepng").decode()] * n, "format": "webp"}


@pytest.fixture
def client():
    register(ModelSpec(key="stub-chat", family="llm", repo_id="stub/chat", engine="stub-llm", resident_gib=1.0))
    register(ModelSpec(key="stub-img", family="diffusion", repo_id="stub/img", engine="stub-image", resident_gib=1.0))
    sched = Scheduler(vram_free=lambda: 100.0, vram_total=lambda: 100.0, start_reaper=False)
    app = create_app(sched)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_models_list(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "stub-chat" in ids and "stub-img" in ids


def test_chat(client):
    r = client.post("/v1/chat/completions", json={
        "model": "stub-chat", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "echo:hi"


def test_images_openai_shape(client):
    r = client.post("/v1/images/generations", json={"model": "stub-img", "prompt": "cat", "n": 2, "size": "512x512"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2
    assert base64.b64decode(data[0]["b64_json"]) == b"fakepng"


def test_unknown_model_404(client):
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404


def test_wrong_family_rejected(client):
    r = client.post("/v1/chat/completions", json={"model": "stub-img", "messages": []})
    assert r.status_code == 400


def test_status_and_admin(client):
    client.post("/v1/chat/completions", json={"model": "stub-chat", "messages": [{"role": "user", "content": "x"}]})
    st = client.get("/status").json()
    assert any(b["model"] == "stub-chat" and b["state"] == "ready" for b in st["backends"])
    client.post("/admin/stop/stub-chat")
    st = client.get("/status").json()
    assert any(b["model"] == "stub-chat" and b["state"] == "unloaded" for b in st["backends"])
