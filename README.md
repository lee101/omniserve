<p align="center">
  <a href="https://app.nz"><img src="assets/appnz-logo.png" alt="app.nz" height="72"></a>
</p>

<h1 align="center">omniserve</h1>

<p align="center"><b>The omni model server.</b> One GPU, every model family: diffusion, video, LoRA, LLM — auto-loaded, VRAM-balanced, OpenAI-compatible.</p>

<p align="center">
  <a href="https://app.nz/deploy?image=ghcr.io/lee101/omniserve:latest&name=omniserve&hardware=gpu-rtx4090&minVramGb=24&idleSeconds=300"><img src="assets/deploy-button.svg" alt="Deploy to app.nz"></a>
</p>

Think AUTOMATIC1111's "serve everything" flexibility with a production server's discipline: a declarative model catalog, a VRAM-aware scheduler that loads/sleeps/evicts models on demand, hot-swappable LoRAs, tiered idle shutdown, and a fleet-friendly weight mirror — all behind OpenAI-compatible HTTP and packaged as a [Cog](https://github.com/replicate/cog) for one-click GPU deploys.

## Why

Replicate and fal give you one endpoint per model. ComfyUI/A1111 give you every model but no server story. omniserve gives you both:

- **Auto-loading** — first request for a model loads it; the scheduler evicts least-recently-used models when VRAM runs short, with capacity preflight before any download.
- **Balancing** — diffusion pipelines, an LTX video pipeline, and vLLM subprocesses coexist on one GPU. LLMs sleep (weights → pinned RAM, VRAM freed, wake in seconds) instead of dying; cold ones unload entirely.
- **LoRA native** — any request can attach LoRAs by catalog id or https URL; adapters download once per fleet via the built-in peer mirror.
- **OpenAI-compatible** — `/v1/chat/completions` (streaming), `/v1/completions`, `/v1/images/generations`, plus `/v1/video/generations` and admin/status routes.
- **Priority tiers** — requests carry `X-Omniserve-Tier: paid | sub | free | background`. GPU slots are granted strictly by tier (FIFO within a tier); `background` runs only on an otherwise idle GPU. Eviction is tier-protected: a free request cannot evict a model a paid caller used in the last `OMNISERVE_TIER_PROTECT_S` (default 120s) — it gets a clean 507 to retry instead — while paid/sub always evict downward. Queue waits past `OMNISERVE_ADMISSION_TIMEOUT_S` return 503 + Retry-After. `/status` reports per-tier active/waiting/served counts.
- **Weight mirror aware** — weights resolve local dir → your R2/HTTP mirror (parallel, range-resume) → Hugging Face.

## Quickstart

```bash
pip install "omniserve[diffusion,llm] @ git+https://github.com/lee101/omniserve"
omniserve serve --port 8000 --preload z-image-turbo
```

```bash
curl localhost:8000/v1/images/generations -d '{"prompt": "a cabin in a snowstorm", "model": "z-image-turbo"}'

curl localhost:8000/v1/chat/completions -d '{"model": "qwen3-4b-instruct", "messages": [{"role": "user", "content": "hi"}]}'

curl localhost:8000/v1/video/generations -d '{"model": "ltx-2.3-distilled", "prompt": "drone shot over a reef", "loras": [{"url": "https://huggingface.co/.../adapter.safetensors", "scale": 0.8}]}'
```

Both models above fit a single 24 GB card together; ask for a third that doesn't fit and the scheduler evicts the least-recently-used one first.

## Catalog

Built-ins: `z-image-turbo`, `flux-schnell`, `sdxl-turbo`, `qwen-image`, `ltx-2.3-distilled`, `qwen3-4b-instruct`, `qwen3-32b`, `gemma-3-12b-it`. List with `omniserve models`.

Add your own with a JSON file (`OMNISERVE_CATALOG=models.json`):

```json
[{"key": "my-flux-finetune", "family": "diffusion", "repo_id": "me/my-flux", "engine": "diffusers",
  "pipeline_class": "FluxPipeline", "steps": 4, "resident_gib": 24, "supports_lora": true}]
```

Engines: `diffusers` (any Diffusers pipeline, torchao fp8/int8/int4 quant, model/sequential offload, optional torch.compile), `vllm` (subprocess with sleep-mode, prefix caching, per-model launch overrides via `extra.cmd`), `ltx` (LTX-2.3 distilled video with LoRA rebuild swap). Adding an engine is one class:

```python
from omniserve.backends.base import Backend, register_engine

@register_engine("my-engine")
class MyBackend(Backend):
    def load(self): ...
    def unload(self): ...
    def infer(self, request: dict) -> dict: ...
```

## Scheduling model

| state | VRAM | wake cost | trigger |
|---|---|---|---|
| ready | resident | — | request |
| sleeping | freed (weights in pinned RAM) | seconds | idle > `OMNISERVE_IDLE_SLEEP_S` (300) |
| unloaded | freed | full load | idle > `OMNISERVE_IDLE_UNLOAD_S` (3600) or LRU eviction |

Requests that can never fit (`resident_gib` > total VRAM) fail fast with HTTP 507 instead of OOMing the box. OOM mid-inference triggers evict-others-and-retry once.

## Fleet weights

```bash
OMNISERVE_MODELS_BASE=https://appstatic.app.nz/models   # manifest.json + files, range-resume, 12-way parallel
OMNISERVE_LORA_MIRROR=http://10.0.0.5:7791              # peer cache: fleet downloads each adapter once
omniserve lora-mirror --port 7791                        # run the peer on any box with the cache
```

## As a Cog

```bash
cog predict -i task=image -i prompt="a red panda astronaut"
cog predict -i task=chat -i model=qwen3-4b-instruct -i prompt="explain KV cache"
cog predict -i task=video -i prompt="waves at sunset" -i lora=<catalog-id-or-url>
```

`cog.yaml` builds one image that serves all three families; `predict.py` shares the same scheduler, so consecutive predictions reuse resident models.

## Deploy on app.nz

One click: [![Deploy to app.nz](assets/deploy-button.svg)](https://app.nz/deploy?image=ghcr.io/lee101/omniserve:latest&name=omniserve&hardware=gpu-rtx4090&minVramGb=24&idleSeconds=300) — scale-to-zero GPU hosting with per-second billing. For LTX video pick an [H100](https://app.nz/deploy?image=ghcr.io/lee101/omniserve:latest&name=omniserve&hardware=gpu-h100&minVramGb=80&idleSeconds=300), or use the built-in [omniserve template](https://app.nz/deploy?template=omniserve).

## Env reference

| var | default | |
|---|---|---|
| `OMNISERVE_CATALOG` | — | extra models JSON |
| `OMNISERVE_HEADROOM_GIB` | 2 | VRAM kept free above model estimate |
| `OMNISERVE_IDLE_SLEEP_S` / `OMNISERVE_IDLE_UNLOAD_S` | 300 / 3600 | idle tiers |
| `OMNISERVE_QUANT` | per-model | `torchao-fp8dq`, `torchao-int8wo`, `fp8-scaled-mm`, … |
| `OMNISERVE_OFFLOAD` | per-model | `cuda`, `model`, `sequential` |
| `OMNISERVE_COMPILE` | off | torch.compile mode for the denoiser |
| `OMNISERVE_MODELS_BASE` | appstatic.app.nz/models | weight mirror |
| `OMNISERVE_LORA_CATALOG` / `OMNISERVE_LORA_MIRROR` | — | LoRA catalog JSON / peer cache |
| `WEIGHTS_DIR` | `/weights` or `/runpod-volume/models` | weight store |
| `HF_TOKEN` / `CIVITAI_TOKEN` | — | gated downloads |

## Development

```bash
uv venv && uv pip install -e '.[dev]'
pytest        # unit tests run without a GPU
```

Apache-2.0. Built by [app.nz](https://app.nz).
