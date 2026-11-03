"""``tle-liquid``: up | calibrate | doctor | fetch | serve | smoke."""

from __future__ import annotations

import argparse
import logging
import sys


def _cfg(a):
    from .config import load_config

    return load_config(
        a.config,
        device=a.device,
        host=getattr(a, "host", None),
        port=getattr(a, "port", None),
        api_key=getattr(a, "api_key", None),
        roles=a.roles,
        max_batch=getattr(a, "max_batch", None),
        lazy_load=True if getattr(a, "lazy", False) else None,
    )


def cmd_doctor(a) -> int:
    from .devices import detect

    try:
        import torch
        import transformers

        print(
            f"torch {torch.__version__} (cuda {torch.version.cuda}, hip {getattr(torch.version, 'hip', None)}), transformers {transformers.__version__}"
        )
        print("device:", detect(a.device or "auto").describe())
    except ImportError as e:
        print("missing dependency:", e, "\nrun packages/tle-liquid/scripts/install.sh")
        return 1
    except RuntimeError as e:
        print(e)
        return 1
    for mod in ("xgrammar", "starlette", "uvicorn", "huggingface_hub"):
        try:
            __import__(mod)
            print(f"{mod}: ok")
        except ImportError:
            print(f"{mod}: MISSING")
    cfg = _cfg(a)
    for r, s in cfg.specs.items():
        print(f"{r}: {s.repo} ({s.kind}, about {s.approx_gb} GB in bf16)")
    return 0


def cmd_fetch(a) -> int:
    from .config import fetch

    cfg = _cfg(a)
    for r in cfg.roles:
        print(f"{r}: downloading {cfg.specs[r].repo} ...", flush=True)
        print("   ->", fetch(cfg.specs[r], a.local_dir))
    return 0


def cmd_serve(a) -> int:
    from .engine import LiquidEngine
    from .server import serve

    cfg = _cfg(a)
    if cfg.host not in ("127.0.0.1", "localhost", "::1") and not cfg.api_key:
        print(
            "warning: listening on",
            cfg.host,
            "without an API key (set TLE_LIQUID_API_KEY or --api-key)",
            file=sys.stderr,
        )
    serve(LiquidEngine(cfg, mode=a.mode))
    return 0


def cmd_up(a) -> int:
    from .up import run

    return run(
        tuple(a.roles or ("s1", "s2")),
        port=a.port or 8100,
        share=a.share,
        layout=a.layout,
        max_batch=a.max_batch,
        key=a.key,
        dry_run=a.dry_run,
        mode=a.mode,
    )


def cmd_calibrate(a) -> int:
    from . import calibrate

    return calibrate.run(
        tuple(a.roles or ("s1", "s2")), layout=a.layout, effort="quick" if a.quick else "full", mode=a.mode
    )


def cmd_smoke(a) -> int:
    from .smoke import run

    return run(_cfg(a), mode=a.mode, game=a.game, n_requests=a.n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tle-liquid", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, serving=False):
        s.add_argument("--config", help="TOML (default config/liquid.toml)")
        s.add_argument("--device", choices=["auto", "cuda", "rocm", "cpu"])
        s.add_argument("--roles", nargs="+", choices=["s1", "s2"], help="roles to load (default both)")
        if serving:
            s.add_argument(
                "--mode",
                choices=["real", "tiny", "mock"],
                default="real",
                help="tiny/mock: no downloads (tests)",
            )
            s.add_argument("--max-batch", type=int)
            s.add_argument("--lazy", action="store_true", help="load a role when its first request arrives")

    s = sub.add_parser("doctor", help="what GPU / libraries / models this machine has")
    common(s)
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("fetch", help="download the weights")
    common(s)
    s.add_argument("--local-dir")
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("serve", help="start the OpenAI-compatible + decision server")
    common(s, True)
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser(
        "up", help="detect this machine, start S1/S2 with the right layout, print ONE link for the game"
    )
    s.add_argument("--roles", nargs="+", choices=["s1", "s2"], help="roles to serve here (default both)")
    s.add_argument("--port", type=int, help="public port of the link (default 8100)")
    s.add_argument(
        "--share", action="store_true", help="listen on the network; the link carries an access key"
    )
    s.add_argument("--layout", choices=["auto", "together", "s1-gpu", "s2-gpu", "cpu"], default="auto")
    s.add_argument(
        "--max-batch", type=int, help="requests fused per model call (default depends on the layout)"
    )
    s.add_argument("--key", help="access key for --share (default: random)")
    s.add_argument("--dry-run", action="store_true", help="only print the detected layout")
    s.add_argument("--mode", choices=["real", "mock"], default="real", help="mock: no weights (tests)")
    s.set_defaults(fn=cmd_up)
    s = sub.add_parser(
        "calibrate", help="find the best G and R for this machine with the game's real requests; saves them"
    )
    s.add_argument("--roles", nargs="+", choices=["s1", "s2"])
    s.add_argument("--layout", choices=["auto", "together", "s1-gpu", "s2-gpu", "cpu"], default="auto")
    s.add_argument("--quick", action="store_true", help="a small grid (minutes instead of tens of minutes)")
    s.add_argument("--mode", choices=["real", "mock"], default="real", help="mock: no weights (tests)")
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser(
        "smoke", help="integrated smoke test (starts a server, fires batched requests, optional game turns)"
    )
    common(s, True)
    s.add_argument("--game", action="store_true", help="also play a few game turns through the server")
    s.add_argument("-n", type=int, default=32, help="concurrent requests per role")
    s.set_defaults(fn=cmd_smoke)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
