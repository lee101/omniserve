#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-ghcr.io/lee101/omniserve:latest}"
PORT="${PORT:-5099}"
WEIGHTS="${WEIGHTS:-/sdb-disk/omniserve-weights}"
NAME=omniserve-test

mkdir -p "$WEIGHTS"
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME --gpus all -p $PORT:5000 \
  -v "$WEIGHTS":/weights -e WEIGHTS_DIR=/weights \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  "$IMAGE"

for i in $(seq 1 90); do
  curl -sf "localhost:$PORT/health-check" >/dev/null 2>&1 && break
  sleep 2
done
curl -s "localhost:$PORT/health-check"; echo

run_predict() {
  curl -s --max-time 3600 -X POST "localhost:$PORT/predictions" \
    -H 'Content-Type: application/json' -d "$1" >/tmp/omniserve_pred.json
  python3 - <<'EOF'
import base64, json
r = json.load(open("/tmp/omniserve_pred.json"))
print("status:", r.get("status"))
if r.get("error"):
    print("error:", str(r["error"])[:500])
out = r.get("output")
if isinstance(out, str) and out.startswith("data:"):
    raw = base64.b64decode(out.split(",", 1)[1])
    open("/tmp/omniserve_out.bin", "wb").write(raw)
    print("output bytes:", len(raw))
else:
    print("output:", str(out)[:300])
EOF
}

echo "--- image (sdxl-turbo) ---"
time run_predict '{"input": {"task": "image", "model": "sdxl-turbo", "prompt": "a neon cyberpunk fox, studio lighting", "width": 512, "height": 512}}'

echo "--- chat (qwen3-4b) ---"
time run_predict '{"input": {"task": "chat", "model": "qwen3-4b-instruct", "prompt": "Reply with exactly: omniserve online"}}'

echo "--- container log tail ---"
docker logs --tail 30 $NAME 2>&1
