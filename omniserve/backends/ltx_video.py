from __future__ import annotations

import base64
import gc
import logging
import os
import tempfile
from pathlib import Path

from .base import Backend, register_engine
from ..weights import resolve_weights

log = logging.getLogger("omniserve.ltx")


@register_engine("ltx")
class LtxVideoBackend(Backend):
    def __init__(self, spec):
        super().__init__(spec)
        self.pipe = None
        self.active_key: tuple = ()

    def load(self) -> None:
        import torch
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ckpt = resolve_weights(self.spec.repo_id, self.spec.single_file)
        gemma = resolve_weights(self.spec.extra.get("gemma_repo", "google/gemma-3-12b-it-qat-q4_0-unquantized"))
        upsampler = self.spec.extra.get("upsampler_file", "spatial-upscaler-x2-1.1.safetensors")
        upsampler_path = resolve_weights(self.spec.repo_id, upsampler)

        kind = os.environ.get("OMNISERVE_QUANT", self.spec.recommended_quant or "fp8-scaled-mm")
        self.quant = None if kind in ("", "none") else QuantizationKind(kind).to_policy(str(ckpt))
        self.offload = OffloadMode(os.environ.get("OMNISERVE_OFFLOAD_LTX", "none"))
        self.ckpt = str(ckpt)
        self.pipe = DistilledPipeline(
            distilled_checkpoint_path=str(ckpt),
            gemma_root=str(gemma),
            spatial_upsampler_path=str(upsampler_path),
            loras=(),
            quantization=self.quant,
            offload_mode=self.offload,
        )
        self.active_key = ()

    def unload(self) -> None:
        self.pipe = None
        self.active_key = ()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _apply_loras(self, specs: list[tuple[str, float]]) -> None:
        key = tuple(specs)
        if key == self.active_key:
            return
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_pipelines.stages import DiffusionStage
        from ltx_pipelines.utils.constants import LTXV_LORA_COMFY_RENAMING_MAP

        loras = tuple(LoraPathStrengthAndSDOps(str(p), s, LTXV_LORA_COMFY_RENAMING_MAP) for p, s in specs)
        self.pipe.stage = DiffusionStage.from_checkpoint(
            self.ckpt, self.pipe.dtype, self.pipe.device,
            loras=loras, quantization=self.quant, offload_mode=self.offload,
        )
        self.active_key = key

    def infer(self, request: dict) -> dict:
        prompt = request.get("prompt", "")
        num_frames = int(request.get("num_frames", 121))
        num_frames = max(9, (num_frames - 1) // 8 * 8 + 1)
        fps = float(request.get("frame_rate", 24.0))
        width = int(request.get("width", 1216)) // 64 * 64
        height = int(request.get("height", 704)) // 64 * 64
        seed = request.get("seed")

        loras = [(lora["path"], float(lora.get("scale", 1.0))) for lora in request.get("loras") or []]
        self._apply_loras(loras)

        out = Path(tempfile.mkdtemp(prefix="omniserve-ltx-")) / "out.mp4"
        self.pipe(
            prompt=prompt, output_path=str(out),
            num_frames=num_frames, frame_rate=fps,
            width=width, height=height,
            seed=int(seed) if seed is not None else None,
            image_path=request.get("image_path"),
        )
        data = out.read_bytes()
        return {"video_b64": base64.b64encode(data).decode(), "format": "mp4", "path": str(out)}
