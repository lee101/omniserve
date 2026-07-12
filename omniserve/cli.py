from __future__ import annotations

import argparse
import json
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="omniserve")
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the omni server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--preload", default="", help="comma-separated model keys to load at boot")

    sub.add_parser("models", help="list catalog")

    mirror = sub.add_parser("lora-mirror", help="run the peer LoRA cache mirror")
    mirror.add_argument("--port", type=int, default=7791)

    pf = sub.add_parser("preflight", help="check capacity for a model without downloading")
    pf.add_argument("model")

    args = p.parse_args()

    if args.cmd == "models":
        from .catalog import MODEL_CATALOG, init_catalog_from_env
        init_catalog_from_env()
        print(json.dumps({k: s.to_dict() for k, s in MODEL_CATALOG.items()}, indent=2))
    elif args.cmd == "preflight":
        from .catalog import get_model, init_catalog_from_env
        from .gpu import free_disk_gib, free_vram_gib, total_vram_gib
        init_catalog_from_env()
        spec = get_model(args.model)
        report = {
            "model": spec.key,
            "needs_disk_gib": spec.download_gib,
            "free_disk_gib": round(free_disk_gib(), 1),
            "needs_vram_gib": spec.resident_gib,
            "free_vram_gib": round(free_vram_gib(), 1),
            "total_vram_gib": round(total_vram_gib(), 1),
        }
        report["fits_now"] = report["free_vram_gib"] >= spec.resident_gib
        report["fits_after_evict"] = report["total_vram_gib"] >= spec.resident_gib
        print(json.dumps(report, indent=2))
    elif args.cmd == "lora-mirror":
        from .loras import serve_mirror
        print(f"lora mirror on :{args.port}")
        serve_mirror(args.port)
    elif args.cmd == "serve":
        import uvicorn
        from .server import create_app
        app = create_app()
        if args.preload:
            for key in args.preload.split(","):
                key = key.strip()
                if key:
                    app.state.scheduler.ensure(key)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
