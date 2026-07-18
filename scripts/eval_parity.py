#!/usr/bin/env python3
"""Quality parity gate: omniserve vs the current production servers.

No cutover happens until, per model, BOTH hold:
  images: mean pixel/embedding similarity across fixed-seed prompts >= threshold
  llm:    judge-scored parity (via an openpaths judge model) >= threshold

Usage:
  python eval_parity.py images --candidate http://localhost:8000 \
      --reference http://127.0.0.1:8100 --model z-image-turbo --out runs/parity
  python eval_parity.py llm --candidate http://localhost:8000 \
      --reference http://127.0.0.1:8300 --model gemma-roleplay-v2 \
      --judge-base https://openpaths.io/v1 --judge-key $OPENPATHS_KEY

Fixed seeds + fixed prompts make image comparisons deterministic where the
backends honor seeding; LLM parity is judged pairwise (win/tie/loss) because
token-identity across stacks is not a realistic bar.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import urllib.request

IMAGE_PROMPTS = [
    "a lighthouse on a cliff at golden hour, oil painting",
    "macro photo of a dew drop on a fern leaf",
    "isometric cutaway of a cozy underground library",
    "a red panda wearing a scarf, studio portrait",
]

LLM_PROMPTS = [
    "Write a 3-sentence opening for a mystery set on a night train.",
    "Explain gradient checkpointing to a new ML engineer in one paragraph.",
    "You are a dungeon master. A player tries to bribe the gate guard with a turnip. Respond in character.",
    "Summarize the tradeoffs of speculative decoding in 4 bullet points.",
]

JUDGE_PROMPT = """Compare two AI responses to the same prompt. Score which is better
for a production chat product (helpfulness, style, coherence). Reply with exactly
one word: A, B, or TIE.

Prompt: {prompt}

Response A:
{a}

Response B:
{b}"""


def post_json(url: str, body: dict, headers: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gen_image(base: str, model: str, prompt: str, seed: int) -> bytes:
    out = post_json(f"{base}/v1/images/generations",
                    {"model": model, "prompt": prompt, "seed": seed, "n": 1})
    item = out["data"][0]
    return base64.b64decode(item["b64_json"])


def image_similarity(a: bytes, b: bytes) -> float:
    """Mean cosine similarity of downsampled luma; 1.0 = identical layout/content."""
    from PIL import Image
    import numpy as np

    def vec(raw):
        img = Image.open(io.BytesIO(raw)).convert("L").resize((64, 64))
        v = np.asarray(img, dtype=np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        return v / n if n else v

    return float((vec(a) * vec(b)).sum())


def run_images(args) -> dict:
    os.makedirs(args.out, exist_ok=True)
    sims = []
    for i, prompt in enumerate(IMAGE_PROMPTS):
        cand = gen_image(args.candidate, args.model, prompt, seed=1000 + i)
        ref = gen_image(args.reference, args.model, prompt, seed=1000 + i)
        for name, raw in (("cand", cand), ("ref", ref)):
            with open(os.path.join(args.out, f"img{i}-{name}.png"), "wb") as f:
                f.write(raw)
        sim = image_similarity(cand, ref)
        sims.append(sim)
        print(f"[{i}] sim={sim:.3f}  {prompt[:50]}")
    mean = statistics.mean(sims)
    verdict = "PASS" if mean >= args.threshold else "FAIL"
    print(f"image parity: mean={mean:.3f} threshold={args.threshold} -> {verdict}")
    return {"mean_similarity": mean, "pass": mean >= args.threshold, "sims": sims}


def chat(base: str, model: str, prompt: str, key: str = "") -> str:
    headers = {"authorization": f"Bearer {key}"} if key else {}
    out = post_json(f"{base}/v1/chat/completions",
                    {"model": model, "max_tokens": 300,
                     "messages": [{"role": "user", "content": prompt}]}, headers)
    return out["choices"][0]["message"]["content"]


def run_llm(args) -> dict:
    wins = ties = losses = 0
    for i, prompt in enumerate(LLM_PROMPTS):
        cand = chat(args.candidate, args.model, prompt)
        ref = chat(args.reference, args.model, prompt)
        # judge twice with swapped order to cancel position bias
        v1 = chat(args.judge_base, args.judge_model,
                  JUDGE_PROMPT.format(prompt=prompt, a=cand, b=ref), args.judge_key).strip().upper()
        v2 = chat(args.judge_base, args.judge_model,
                  JUDGE_PROMPT.format(prompt=prompt, a=ref, b=cand), args.judge_key).strip().upper()
        cand_score = (v1.startswith("A")) + (v2.startswith("B"))
        if cand_score == 2:
            wins += 1
        elif cand_score == 0:
            losses += 1
        else:
            ties += 1
        print(f"[{i}] {['loss','tie','win'][cand_score]}  {prompt[:50]}")
    parity = (wins + ties) / len(LLM_PROMPTS)
    verdict = "PASS" if parity >= args.threshold else "FAIL"
    print(f"llm parity: win+tie={parity:.2f} threshold={args.threshold} -> {verdict}")
    return {"wins": wins, "ties": ties, "losses": losses, "pass": parity >= args.threshold}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("images", "llm"):
        p = sub.add_parser(name)
        p.add_argument("--candidate", required=True, help="omniserve base URL")
        p.add_argument("--reference", required=True, help="current prod base URL")
        p.add_argument("--model", required=True)
        p.add_argument("--out", default="runs/parity")
    sub.choices["images"].add_argument("--threshold", type=float, default=0.85)
    llm = sub.choices["llm"]
    llm.add_argument("--threshold", type=float, default=0.75, help="min win+tie rate")
    llm.add_argument("--judge-base", default="https://openpaths.io/v1")
    llm.add_argument("--judge-model", default="claude-sonnet-5")
    llm.add_argument("--judge-key", default=os.environ.get("OPENPATHS_KEY", ""))
    args = ap.parse_args()

    result = run_images(args) if args.cmd == "images" else run_llm(args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.cmd}-{args.model.replace('/', '_')}.json"), "w") as f:
        json.dump(result, f, indent=2)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
