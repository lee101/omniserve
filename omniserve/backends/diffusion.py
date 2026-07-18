from __future__ import annotations

import base64
import gc
import io
import logging
import os

from .base import Backend, register_engine
from ..weights import hf_cache_dir, resolve_weights

log = logging.getLogger("omniserve.diffusion")


def _quant_config(name: str):
    if not name or not name.startswith("torchao-"):
        return None
    from diffusers import PipelineQuantizationConfig, TorchAoConfig
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
    )
    configs = {
        "torchao-fp8dq": Float8DynamicActivationFloat8WeightConfig,
        "torchao-fp8wo": Float8WeightOnlyConfig,
        "torchao-int8wo": Int8WeightOnlyConfig,
        "torchao-int4wo": Int4WeightOnlyConfig,
    }
    if name not in configs:
        return None
    return PipelineQuantizationConfig(quant_mapping={"transformer": TorchAoConfig(configs[name]())})


@register_engine("diffusers")
class DiffusionBackend(Backend):
    def __init__(self, spec):
        super().__init__(spec)
        self.pipe = None
        self.active_loras: tuple = ()

    def load(self) -> None:
        import torch
        import diffusers

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            src = str(resolve_weights(self.spec.repo_id, self.spec.single_file, allow_hf=False))
        except FileNotFoundError:
            src = self.spec.repo_id
        cls = getattr(diffusers, self.spec.pipeline_class, diffusers.DiffusionPipeline)
        kwargs: dict = {"torch_dtype": torch.bfloat16}
        if src == self.spec.repo_id:
            kwargs["cache_dir"] = str(hf_cache_dir())
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
        quant = os.environ.get("OMNISERVE_QUANT", self.spec.recommended_quant)
        qc = _quant_config(quant)
        if qc is not None:
            kwargs["quantization_config"] = qc

        if self.spec.loader == "single_file" or self.spec.single_file:
            pipe = cls.from_single_file(src, **kwargs)
        else:
            variant = self.spec.extra.get("variant")
            if variant:
                kwargs["variant"] = variant
            try:
                pipe = cls.from_pretrained(src, **kwargs)
            except (OSError, EnvironmentError):
                if kwargs.get("variant"):
                    raise
                log.info("retrying %s with fp16 variant weights", self.spec.key)
                pipe = cls.from_pretrained(src, variant="fp16", **kwargs)

        offload = os.environ.get("OMNISERVE_OFFLOAD", self.spec.recommended_offload)
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")

        compile_mode = os.environ.get("OMNISERVE_COMPILE", "")
        if compile_mode:
            target = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
            if target is not None:
                compiled = torch.compile(target, mode=compile_mode, dynamic=True)
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    pipe.transformer = compiled
                else:
                    pipe.unet = compiled
        self.pipe = pipe

    def unload(self) -> None:
        self.pipe = None
        self.active_loras = ()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _apply_loras(self, loras: list[dict]) -> None:
        key = tuple((l["path"], float(l.get("scale", 1.0))) for l in loras)
        if key == self.active_loras:
            return
        if self.active_loras:
            self.pipe.unload_lora_weights()
        names = []
        for i, (path, scale) in enumerate(key):
            name = f"lora_{i}"
            self.pipe.load_lora_weights(path, adapter_name=name)
            names.append((name, scale))
        if names:
            self.pipe.set_adapters([n for n, _ in names], [s for _, s in names])
        self.active_loras = key

    def infer(self, request: dict) -> dict:
        import torch

        prompt = request.get("prompt", "")
        n = int(request.get("n", 1))
        width = int(request.get("width", 1024))
        height = int(request.get("height", 1024))
        steps = int(request.get("steps", self.spec.steps or 20))
        guidance = request.get("guidance_scale")
        seed = request.get("seed")

        loras = request.get("loras") or []
        if loras and self.spec.supports_lora:
            self._apply_loras(loras)
        elif self.active_loras:
            self._apply_loras([])

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cuda").manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt, width=width - width % 16, height=height - height % 16,
            num_inference_steps=steps, num_images_per_prompt=n, generator=gen,
        )
        if guidance is not None:
            kwargs["guidance_scale"] = float(guidance)
        neg = request.get("negative_prompt")
        if neg:
            kwargs["negative_prompt"] = neg

        with torch.inference_mode():
            out = self.pipe(**kwargs)

        images = []
        for img in out.images:
            buf = io.BytesIO()
            img.save(buf, format=request.get("format", "WEBP").upper())
            images.append(base64.b64encode(buf.getvalue()).decode())
        return {"images_b64": images, "format": request.get("format", "webp")}
