from __future__ import annotations

import base64
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import backends  # noqa: F401
from .catalog import MODEL_CATALOG, get_model, init_catalog_from_env
from .loras import ensure_lora, load_lora_catalog
from .scheduler import CapacityError, Scheduler

log = logging.getLogger("omniserve")


def _default_model(family: str) -> str:
    env = os.environ.get(f"OMNISERVE_DEFAULT_{family.upper()}")
    if env:
        return env
    for spec in MODEL_CATALOG.values():
        if spec.family == family:
            return spec.key
    raise HTTPException(400, f"no {family} model in catalog")


def _resolve(model: str | None, family: str) -> str:
    if not model or model in ("auto", "default"):
        return _default_model(family)
    if model in MODEL_CATALOG:
        return model
    for key, spec in MODEL_CATALOG.items():
        if spec.repo_id == model or spec.repo_id.split("/")[-1].lower() == model.lower():
            return key
    raise HTTPException(404, f"unknown model '{model}'")


def _prepare_loras(items: list | None) -> list[dict]:
    out = []
    for item in items or []:
        if isinstance(item, str):
            path = ensure_lora(lora_id=item) if not item.startswith("https://") else ensure_lora(url=item)
            out.append({"path": str(path), "scale": 1.0})
        else:
            ref = item.get("id") or item.get("url") or item.get("path")
            if item.get("path"):
                path = item["path"]
            elif str(ref).startswith("https://"):
                path = str(ensure_lora(url=ref))
            else:
                path = str(ensure_lora(lora_id=ref))
            out.append({"path": path, "scale": float(item.get("scale", 1.0))})
    return out


def create_app(scheduler: Scheduler | None = None) -> FastAPI:
    init_catalog_from_env()
    app = FastAPI(title="omniserve", version="0.1.0")
    sched = scheduler or Scheduler()
    app.state.scheduler = sched
    started = time.time()

    @app.get("/health")
    def health():
        return {"status": "ok", "uptime_s": round(time.time() - started, 1)}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": k, "object": "model", "owned_by": "omniserve", **{"family": s.family, "engine": s.engine}}
            for k, s in MODEL_CATALOG.items()
        ]}

    @app.get("/status")
    def status():
        return sched.status()

    @app.post("/admin/ensure/{key}")
    def admin_ensure(key: str):
        sched.ensure(key)
        return sched.status()

    @app.post("/admin/sleep/{key}")
    def admin_sleep(key: str):
        sched.sleep(key)
        return sched.status()

    @app.post("/admin/stop/{key}")
    def admin_stop(key: str):
        sched.stop(key)
        return sched.status()

    @app.get("/loras")
    def loras():
        return load_lora_catalog()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        key = _resolve(body.get("model"), "llm")
        spec = get_model(key)
        if spec.family != "llm":
            raise HTTPException(400, f"'{key}' is a {spec.family} model")
        if body.get("stream"):
            b = sched.ensure(key)
            return StreamingResponse(
                b.proxy_stream("/v1/chat/completions", body),
                media_type="text/event-stream",
            )
        body["_path"] = "/v1/chat/completions"
        return _run(sched, key, body)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        key = _resolve(body.get("model"), "llm")
        if body.get("stream"):
            b = sched.ensure(key)
            return StreamingResponse(
                b.proxy_stream("/v1/completions", body),
                media_type="text/event-stream",
            )
        body["_path"] = "/v1/completions"
        return _run(sched, key, body)

    @app.post("/v1/images/generations")
    async def images(request: Request):
        body = await request.json()
        key = _resolve(body.get("model"), "diffusion")
        size = body.get("size", "1024x1024")
        try:
            w, h = (int(x) for x in size.lower().split("x"))
        except Exception:
            w = h = 1024
        req = {
            "prompt": body.get("prompt", ""),
            "n": int(body.get("n", 1)),
            "width": w, "height": h,
            "steps": body.get("steps"),
            "guidance_scale": body.get("guidance_scale"),
            "negative_prompt": body.get("negative_prompt"),
            "seed": body.get("seed"),
            "loras": _prepare_loras(body.get("loras")),
        }
        req = {k: v for k, v in req.items() if v is not None}
        result = _run(sched, key, req)
        return {
            "created": int(time.time()),
            "data": [{"b64_json": b} for b in result["images_b64"]],
            "model": key,
        }

    @app.post("/v1/video/generations")
    async def video(request: Request):
        body = await request.json()
        key = _resolve(body.get("model"), "video")
        req = {
            "prompt": body.get("prompt", ""),
            "num_frames": body.get("num_frames", 121),
            "frame_rate": body.get("frame_rate", 24),
            "width": body.get("width", 1216),
            "height": body.get("height", 704),
            "seed": body.get("seed"),
            "image_path": body.get("image_path"),
            "loras": _prepare_loras(body.get("loras")),
        }
        result = _run(sched, key, req)
        if body.get("raw"):
            return Response(base64.b64decode(result["video_b64"]), media_type="video/mp4")
        return {"created": int(time.time()), "data": [{"b64_json": result["video_b64"], "format": "mp4"}], "model": key}

    @app.exception_handler(CapacityError)
    def capacity_handler(_req, exc):
        return JSONResponse(status_code=507, content={"error": str(exc)})

    return app


def _run(sched: Scheduler, key: str, req: dict) -> dict:
    try:
        return sched.infer(key, req)
    except CapacityError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed for %s", key)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
