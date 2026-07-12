from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def lora_cache_dir() -> Path:
    for cand in (os.environ.get("OMNISERVE_LORA_CACHE"), "/runpod-volume/loras"):
        if cand and Path(cand).is_dir():
            return Path(cand)
    d = Path(os.environ.get("OMNISERVE_HOME", Path.home() / ".cache/omniserve")) / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_lora_catalog() -> dict[str, dict]:
    path = os.environ.get("OMNISERVE_LORA_CATALOG")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def _auth_headers(url: str) -> dict:
    if "civitai.com" in url and os.environ.get("CIVITAI_TOKEN"):
        return {"Authorization": f"Bearer {os.environ['CIVITAI_TOKEN']}"}
    if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
        return {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    return {}


def _download(url: str, dest: Path) -> Path:
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=_auth_headers(url))
    with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if part.stat().st_size < (1 << 20):
        part.unlink(missing_ok=True)
        raise IOError(f"suspiciously small download from {url}")
    os.replace(part, dest)
    return dest


def ensure_lora(lora_id: str | None = None, url: str | None = None) -> Path:
    catalog = load_lora_catalog()
    if lora_id and lora_id in catalog:
        url = catalog[lora_id]["url"]
        name = f"{lora_id}.safetensors"
    elif url:
        name = urllib.parse.quote(url, safe="") + ".safetensors"
    else:
        raise ValueError("need lora_id or url")
    if not (url or "").startswith("https://"):
        raise ValueError("lora url must be https")

    dest = lora_cache_dir() / name
    if dest.exists():
        return dest

    mirror = os.environ.get("OMNISERVE_LORA_MIRROR")
    if mirror:
        try:
            mirror_url = (f"{mirror}/lora/{lora_id}" if lora_id and lora_id in catalog
                          else f"{mirror}/url?u={urllib.parse.quote(url, safe='')}")
            return _download(mirror_url, dest)
        except Exception:
            pass
    return _download(url, dest)


class _MirrorHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/catalog":
                body = json.dumps(load_lora_catalog()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/lora/"):
                path = ensure_lora(lora_id=parsed.path.split("/lora/", 1)[1])
            elif parsed.path == "/url":
                q = urllib.parse.parse_qs(parsed.query)
                path = ensure_lora(url=q["u"][0])
            else:
                self.send_error(404)
                return
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)
        except Exception as e:
            try:
                self.send_error(500, str(e))
            except Exception:
                pass


def serve_mirror(port: int = 7791) -> None:
    ThreadingHTTPServer(("0.0.0.0", port), _MirrorHandler).serve_forever()
