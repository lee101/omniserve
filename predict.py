from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path as P

from cog import BasePredictor, Input, Path

from omniserve import backends  # noqa: F401
from omniserve.catalog import MODEL_CATALOG, init_catalog_from_env
from omniserve.loras import ensure_lora
from omniserve.scheduler import Scheduler


import logging


class Predictor(BasePredictor):
    def setup(self):
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
        init_catalog_from_env()
        self.scheduler = Scheduler()

    def predict(
        self,
        task: str = Input(default="image", choices=["image", "video", "chat"], description="Inference task"),
        model: str = Input(default="auto", description="Catalog model key, or 'auto' for the family default"),
        prompt: str = Input(default="", description="Prompt (or chat user message)"),
        system_prompt: str = Input(default="", description="System prompt for chat"),
        messages_json: str = Input(default="", description="Full chat messages as JSON, overrides prompt"),
        negative_prompt: str = Input(default="", description="Negative prompt (image)"),
        width: int = Input(default=1024, ge=256, le=2048),
        height: int = Input(default=1024, ge=256, le=2048),
        steps: int = Input(default=0, ge=0, le=100, description="0 = model default"),
        guidance_scale: float = Input(default=0.0, ge=0.0, le=20.0, description="0 = model default"),
        num_frames: int = Input(default=121, ge=9, le=257, description="Video frames, snapped to 8k+1"),
        frame_rate: float = Input(default=24.0, ge=8, le=50),
        lora: str = Input(default="", description="LoRA catalog id or https url"),
        lora_strength: float = Input(default=1.0, ge=0.0, le=2.0),
        max_tokens: int = Input(default=1024, ge=1, le=32768),
        temperature: float = Input(default=0.7, ge=0.0, le=2.0),
        seed: int = Input(default=-1, description="-1 = random"),
    ) -> Path:
        family = {"image": "diffusion", "video": "video", "chat": "llm"}[task]
        key = model
        if key in ("", "auto", "default"):
            key = next(k for k, s in MODEL_CATALOG.items() if s.family == family)

        loras = []
        if lora:
            path = ensure_lora(url=lora) if lora.startswith("https://") else ensure_lora(lora_id=lora)
            loras.append({"path": str(path), "scale": lora_strength})

        out_dir = P(tempfile.mkdtemp(prefix="omniserve-"))

        if task == "chat":
            if messages_json:
                messages = json.loads(messages_json)
            else:
                messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": prompt}]
            req = {
                "_path": "/v1/chat/completions",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            result = self.scheduler.infer(key, req)
            out = out_dir / "response.json"
            out.write_text(json.dumps(result, indent=2))
            return Path(out)

        if task == "video":
            req = {
                "prompt": prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "width": width,
                "height": height,
                "loras": loras,
            }
            if seed >= 0:
                req["seed"] = seed
            result = self.scheduler.infer(key, req)
            out = out_dir / "out.mp4"
            out.write_bytes(base64.b64decode(result["video_b64"]))
            return Path(out)

        req = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": width,
            "height": height,
            "loras": loras,
        }
        if steps:
            req["steps"] = steps
        if guidance_scale:
            req["guidance_scale"] = guidance_scale
        if seed >= 0:
            req["seed"] = seed
        req = {k: v for k, v in req.items() if v is not None}
        result = self.scheduler.infer(key, req)
        out = out_dir / "out.webp"
        out.write_bytes(base64.b64decode(result["images_b64"][0]))
        return Path(out)
