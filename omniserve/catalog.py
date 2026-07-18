from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    repo_id: str
    engine: str
    pipeline_class: str = ""
    task: str = ""
    steps: int = 0
    download_gib: float = 0.0
    resident_gib: float = 0.0
    license: str = ""
    loader: str = "pretrained"
    recommended_offload: str = "cuda"
    recommended_quant: str = ""
    supports_lora: bool = False
    single_file: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_CATALOG: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    MODEL_CATALOG[spec.key] = spec
    return spec


register(ModelSpec(
    key="z-image-turbo", family="diffusion", repo_id="Tongyi-MAI/Z-Image-Turbo",
    engine="diffusers", pipeline_class="ZImagePipeline", task="text-to-image",
    steps=8, download_gib=12.0, resident_gib=13.0, license="apache-2.0",
    recommended_quant="torchao-fp8dq", supports_lora=True,
))
register(ModelSpec(
    key="flux-schnell", family="diffusion", repo_id="black-forest-labs/FLUX.1-schnell",
    engine="diffusers", pipeline_class="FluxPipeline", task="text-to-image",
    steps=4, download_gib=23.0, resident_gib=24.0, license="apache-2.0",
    recommended_offload="model", recommended_quant="torchao-fp8dq", supports_lora=True,
))
register(ModelSpec(
    key="sdxl-turbo", family="diffusion", repo_id="stabilityai/sdxl-turbo",
    engine="diffusers", pipeline_class="AutoPipelineForText2Image", task="text-to-image",
    steps=4, download_gib=7.0, resident_gib=8.5, license="sai-nc",
    supports_lora=True, extra={"variant": "fp16"},
))
register(ModelSpec(
    key="qwen-image", family="diffusion", repo_id="Qwen/Qwen-Image",
    engine="diffusers", pipeline_class="DiffusionPipeline", task="text-to-image",
    steps=30, download_gib=41.0, resident_gib=42.0, license="apache-2.0",
    recommended_offload="model", recommended_quant="torchao-fp8dq", supports_lora=True,
))
register(ModelSpec(
    key="ltx-2.3-distilled", family="video", repo_id="Lightricks/LTX-2.3",
    engine="ltx", task="text-to-video", steps=8,
    download_gib=48.0, resident_gib=60.0, license="ltx-open-weights",
    recommended_quant="fp8-scaled-mm", supports_lora=True,
    single_file="ltx-2.3-22b-distilled-1.1.safetensors",
))
register(ModelSpec(
    key="qwen3-4b-instruct", family="llm", repo_id="Qwen/Qwen3-4B-Instruct-2507",
    engine="vllm", task="chat", download_gib=8.0, resident_gib=12.0,
    license="apache-2.0",
))
register(ModelSpec(
    key="qwen3-32b", family="llm", repo_id="Qwen/Qwen3-32B",
    engine="vllm", task="chat", download_gib=65.0, resident_gib=70.0,
    license="apache-2.0", recommended_quant="fp8",
))
register(ModelSpec(
    key="gemma-3-12b-it", family="llm", repo_id="google/gemma-3-12b-it",
    engine="vllm", task="chat", download_gib=24.0, resident_gib=28.0,
    license="gemma",
))


def load_extra_catalog(path: str | Path) -> int:
    data = json.loads(Path(path).read_text())
    n = 0
    for item in data if isinstance(data, list) else data.get("models", []):
        register(ModelSpec(**item))
        n += 1
    return n


def register_proxy_defaults() -> None:
    """Register proxy catalog entries for each modality that has an upstream
    configured via OMNISERVE_PROXY_<KEY>. Lets one omniserve front an existing
    fleet (image/LLM/TTS/STT) without re-hosting any model. Only entries whose
    base_url env is set are registered, so this is a no-op on a fresh box."""
    proxies = [
        ("proxy-llm", "llm", "chat"),
        ("proxy-image", "diffusion", "text-to-image"),
        ("proxy-tts", "tts", "text-to-speech"),
        ("proxy-stt", "stt", "speech-to-text"),
    ]
    for key, family, task in proxies:
        env_key = "OMNISERVE_PROXY_" + key.upper().replace("-", "_")
        base = os.environ.get(env_key)
        if not base:
            continue
        register(ModelSpec(
            key=key, family=family, repo_id=key, engine="proxy", task=task,
            resident_gib=0.0,
            extra={"base_url": base,
                   "model_override": os.environ.get(env_key + "_MODEL", "")},
        ))


def init_catalog_from_env() -> None:
    extra = os.environ.get("OMNISERVE_CATALOG")
    if extra and Path(extra).exists():
        load_extra_catalog(extra)
    register_proxy_defaults()


def get_model(key: str) -> ModelSpec:
    if key not in MODEL_CATALOG:
        raise KeyError(f"unknown model '{key}'; known: {sorted(MODEL_CATALOG)}")
    return MODEL_CATALOG[key]


def models_for_family(family: str) -> list[ModelSpec]:
    return [s for s in MODEL_CATALOG.values() if s.family == family]
