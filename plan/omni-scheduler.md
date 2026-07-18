# Priority scheduling for a shared GPU

Goal: one server owns a GPU (or a small pool) for several independent
workloads — diffusion, video, LLM chat/completion — instead of N processes that
each assume the whole card and only coexist by luck.

## Priority economy

Every request carries a tier (header `X-Omniserve-Tier`, set by the caller's
gateway, which knows the request's billing/priority class):

| tier | typical source | policy |
|---|---|---|
| `paid` | metered/paid API calls | always first; may evict anything |
| `sub` | flat-rate subscription traffic | ahead of free; may evict free/background |
| `free` | unauthenticated demos, trials | queued behind paid+sub; cannot evict recently-used higher-tier models (507 instead) |
| `background` | "run while the GPU is idle" jobs (batch, evals, index builds) | admitted only when nothing else runs or waits; first to be starved |

Scheduling rules (in `priority.py` + `scheduler.py`):
- Admission is a priority queue over GPU slots; among equal tiers, FIFO.
- Tier-protected eviction: a lower tier cannot evict a model a higher tier used
  within the protection window (`OMNISERVE_TIER_PROTECT_S`, default 120s).
  Paid/sub always evict downward.
- Background runs only on an otherwise-idle GPU and yields between requests.
- OOM recovery: evict-others-and-retry; requests that still cannot fit fail
  clean (507) instead of wedging the card — recoverable by client retry.

## Offlining / elasticity

- Models load on demand, sleep after `OMNISERVE_IDLE_SLEEP_S` (LLM weights →
  pinned RAM, VRAM freed, wake in seconds), unload after
  `OMNISERVE_IDLE_UNLOAD_S`.
- Burst offlining: a paid batch (e.g. several images at once) that needs VRAM
  evicts lower-tier residents, which lazy-reload afterward — replacing manual
  unload coordination between separate servers.

## Integration

Front each existing workload with an OpenAI-compatible adapter and set the tier
header from its gateway's view of the caller:

- image server → `/v1/images/generations`
- LLM/completions adapter → `/v1/chat/completions` + `/v1/completions`
- any provider passthrough → OpenAI-compatible, tier header from billing state

Content moderation, per-tenant auth, and any product-specific pre/post
processing stay in the calling gateway; this server only schedules GPU work.

## Quality gates before cutover

`scripts/eval_parity.py` compares a candidate omniserve deployment against an
existing server: fixed-seed image similarity and judge-scored LLM parity.
Cut over per-model only when parity clears the threshold AND throughput matches
or beats the incumbent under a replayed load trace. Shadow a small % of traffic
first, then canary, then cut over.

## Performance roadmap

1. Now (Python): priority gate, tier-protected eviction, batch endpoints —
   correctness and scheduling; overhead is negligible vs model runtime.
2. Next: continuous batching for diffusion (coalesce same-model image
   requests), pinned-RAM sleep for diffusion pipelines.
3. Later: hot paths in C/CUDA (fused attention/sampling, custom fp8 kernels)
   under the diffusers/vLLM engines where profiles show wins. The HTTP and
   scheduling layers stay Python.
