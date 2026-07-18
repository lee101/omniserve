# omniserve as the shared GPU scheduling environment

Goal: one omni server owns the 32GB 5090 (and any future GPUs) for all
properties — cutedsl.cc, text-generator.io, app.nz, openpaths.io (we are a
provider there) — replacing today's N independent servers that each assume the
whole card and only coexist by luck.

## Priority economy

Every request carries a tier (internal header `X-Omniserve-Tier`, set by each
site's gateway which knows the caller's billing state):

| tier | who | policy |
|---|---|---|
| `paid` | metered/paid entrypoints (openpaths, cutedsl credits, app.nz billed) | always first; may evict anything |
| `sub` | text-generator unlimited subscription | ahead of free; may evict free/background models |
| `free` | unauthenticated demos, trials | queued behind paid+sub; cannot evict recently-used higher-tier models (507 instead) |
| `background` | "while nothing else is running" jobs (evals, distills, index builds) | admitted only when the GPU is otherwise idle; first to be starved |

Scheduling rules (implemented in `priority.py` + `scheduler.py`):
- Admission is a priority queue over GPU slots; among equal tiers FIFO.
- Tier-protected eviction: a lower tier cannot evict a model a higher tier used
  within the protection window (default 120s). Paid/sub can always evict down.
- Background runs only when no other tier is running or waiting, and yields the
  slot between requests.
- OOM recovery stays: evict-others-and-retry, and requests fail clean (507)
  rather than wedging the card — recoverable by client retry.

## Offlining / elasticity

- Models load on demand, sleep (vLLM: weights to pinned RAM) after
  `idle_sleep_s`, unload after `idle_unload_s` — existing.
- Burst offlining: a paid batch (e.g. 4× images) that needs VRAM triggers
  eviction of lower-tier residents; they lazy-reload after. This replaces
  today's manual `/admin/unload` dance between cutedsl's image server and the
  tg vLLM backend.

## Integration map (adapters, no behavior change per site)

- cutedsl.cc image server → `/v1/images/generations` (zimage/chronos catalog
  entries; NSFW classifier stays site-side).
- text-generator.io vLLM parity adapter (:8300) → `/v1/chat/completions` +
  `/v1/completions` with the roleplay catalog entry (fp8 + MTP config via
  `extra.cmd`); autocomplete readiness maps to `/status`.
- app.nz gateway + openpaths provider → OpenAI-compatible passthrough with
  tier header from billing state.

## Quality gates before any cutover

`scripts/eval_parity.py`: fixed-seed image prompts (SSIM/CLIP distance vs
current server outputs) and fixed LLM prompt set (judge-scored parity via
openpaths). Cutover per-model only when parity within threshold AND
throughput >= current under a replayed load trace. Shadow first (mirror small
% of real traffic, compare), then canary, then cutover. No cutover in this
phase.

## Performance roadmap

1. Now (python): priority gate, protected eviction, batch endpoints — correctness
   and scheduling wins, negligible overhead vs model runtime.
2. Next: continuous batching for diffusion (queue-coalesce same-model image
   requests), pinned-RAM sleep for diffusion pipelines (already for vLLM).
3. Later: hot loop in C/CUDA — cutedslkernels for this machine's 5090
   (sm_120): fused attention/sampling paths, custom fp8 kernels. The HTTP/
   scheduling layer stays python; kernels swap in under the diffusers/vllm
   engines where profiles show wins.
4. Open source: repo is already standalone (MIT license file present); publish
   once priority scheduling + parity harness land.
