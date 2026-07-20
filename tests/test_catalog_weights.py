import json

from omniserve.catalog import MODEL_CATALOG, get_model, load_extra_catalog, models_for_family
from omniserve.weights import resolve_weights, weights_dir


def test_builtin_catalog():
    assert "flux-schnell" in MODEL_CATALOG
    spec = get_model("flux-schnell")
    assert spec.family == "diffusion"
    assert spec.resident_gib > 0
    assert models_for_family("llm")


def test_extra_catalog(tmp_path):
    f = tmp_path / "extra.json"
    f.write_text(json.dumps([{
        "key": "my-model", "family": "llm", "repo_id": "me/my-model", "engine": "vllm",
        "resident_gib": 5.0}]))
    assert load_extra_catalog(f) == 1
    assert get_model("my-model").repo_id == "me/my-model"


def test_weights_local_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    d = tmp_path / "org/model"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    assert resolve_weights("org/model") == d
    assert resolve_weights("org/model", "model.safetensors").name == "model.safetensors"


def test_weights_incomplete_not_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    d = tmp_path / "org/partial"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    (d / ".incomplete").touch()
    try:
        resolve_weights("org/partial", allow_download=False)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_weights_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path


def test_weights_no_hf_fallback_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("OMNISERVE_MODELS_BASE", "")
    try:
        resolve_weights("org/never-cached", allow_hf=False)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
