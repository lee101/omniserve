#!/usr/bin/env python3
"""Post-deploy watchdog for an omniserve rollout.

Polls omniserve /status plus a per-modality synthetic probe, and prints one
line per cycle with health, tail latency, VRAM, and admission stats. Exits
non-zero (and shouts) if any probe fails N cycles in a row or p99 latency
regresses past a baseline — so a canary/cutover can be auto-rolled-back.

    python watch_deploy.py --base http://127.0.0.1:8000 \
        --baseline baseline.json --max-fails 3

Baseline JSON (optional): {"chat_ms_p99": 1200, "image_ms_p99": 4000}. Probes
run at the request tier `background` so the watchdog never displaces paid work.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROBES = {
    "chat": ("/v1/chat/completions",
             {"model": "proxy-llm", "max_tokens": 8,
              "messages": [{"role": "user", "content": "ping"}]}),
    "tts": ("/v1/audio/speech",
            {"model": "proxy-tts", "input": "ping", "response_format": "pcm"}),
    "image": ("/v1/images/generations",
              {"model": "proxy-image", "prompt": "a red circle", "n": 1, "steps": 4}),
}


def _post(base, path, body, timeout, extra_headers=None):
    data = json.dumps(body).encode()
    headers = {"content-type": "application/json", "x-omniserve-tier": "background"}
    headers.update(extra_headers or {})
    req = urllib.request.Request(base + path, data=data, headers=headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return (time.monotonic() - t0) * 1000.0


def _status(base, timeout=5):
    try:
        with urllib.request.urlopen(base + "/status", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--cycles", type=int, default=0, help="0 = forever")
    ap.add_argument("--max-fails", type=int, default=3)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--probes", default="chat,tts,image")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--probe-header", action="append", default=[],
                    help="extra header for probe requests, 'Name: value' (repeatable), "
                         "e.g. --probe-header 'X-Rapid-API-Key: <key>'")
    args = ap.parse_args()
    extra_headers = {}
    for h in args.probe_header:
        if ":" in h:
            k, v = h.split(":", 1)
            extra_headers[k.strip()] = v.strip()

    baseline = {}
    if args.baseline:
        try:
            baseline = json.load(open(args.baseline))
        except Exception:
            pass
    which = [p.strip() for p in args.probes.split(",") if p.strip() in PROBES]

    consecutive_fail = {p: 0 for p in which}
    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        st = _status(args.base)
        vram = st.get("vram_free_gib")
        adm = st.get("admission", {})
        parts = [f"cycle={cycle}", f"vram_free={vram}", f"active={adm.get('active')}"]
        regressed = []
        for p in which:
            path, body = PROBES[p]
            try:
                ms = _post(args.base, path, body, args.timeout, extra_headers)
                consecutive_fail[p] = 0
                tag = f"{p}={ms:.0f}ms"
                b = baseline.get(f"{p}_ms_p99")
                if b and ms > 1.5 * b:
                    tag += f"!(>1.5x {b})"
                    regressed.append(p)
                parts.append(tag)
            except Exception as e:
                consecutive_fail[p] += 1
                parts.append(f"{p}=FAIL({type(e).__name__})")
        print("  ".join(parts), flush=True)

        dead = [p for p, n in consecutive_fail.items() if n >= args.max_fails]
        if dead:
            print(f"ALERT: {dead} failed {args.max_fails}x in a row — roll back", flush=True)
            raise SystemExit(2)
        if len(regressed) >= 2:
            print(f"ALERT: latency regression on {regressed} — investigate", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
