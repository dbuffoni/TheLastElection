"""``tle-liquid up``: one command that makes this machine serve S1 and/or S2 and prints ONE link for the game.

1. It looks at the hardware (in a subprocess) and chooses a layout:
   * dedicated GPU (NVIDIA CUDA or a discrete AMD card): one process, both models on the GPU;
   * integrated AMD GPU: S1 on the CPU, S2 on the GPU, two processes. Measured on a Radeon 780M: running both models on the
     iGPU hangs the GPU under game load; S1 on the CPU and S2 on the GPU is stable;
   * no GPU: two CPU processes, the cores split between them.
   One process per model means the two never wait for each other (S1 and S2 run in parallel).
2. A small gateway listens on the public port and forwards ``/api/alpha/decisions`` to S1 and ``/v1/*`` to S2, so a single
   address serves both. With ``--share`` it listens on the network and the address carries a secret path prefix
   (``http://host:8100/k/<key>``): the link alone is enough, nothing else to configure on the main machine.
3. It waits until the models are loaded, then prints the link and the exact variable to export on the machine that plays.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

RESTART_GRACE = 240.0  # seconds a request waits for a crashed backend to come back
LAYOUTS = ("auto", "together", "split-gpu", "cpu")


@dataclass
class Backend:
    roles: tuple[str, ...]
    device: str  # cpu | cuda | rocm
    port: int
    max_batch: int
    threads: int | None = None
    env: dict[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        a = [
            sys.executable,
            "-m",
            "tle.liquid.cli",
            "serve",
            "--roles",
            *self.roles,
            "--device",
            self.device,
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
            "--max-batch",
            str(self.max_batch),
        ]
        return a


def plan(
    hw: dict,
    roles: tuple[str, ...],
    base_port: int,
    layout: str = "auto",
    max_batch: int | None = None,
    batch_by_role: dict[str, int] | None = None,
):
    """Pure function: hardware description -> (layout name, backends). Ports: base_port is the public gateway, the
    backends take the next ones."""
    from .hw import hsa_override_for

    cores = max(1, int(hw.get("cores", 1)))
    gpu = hw.get("backend", "cpu")
    if layout == "auto":
        if gpu == "cpu":
            layout = "cpu"
        elif hw.get("integrated"):
            layout = "split-gpu"
        else:
            layout = "together"
    if layout != "cpu" and gpu == "cpu":
        raise RuntimeError(f"layout '{layout}' needs a GPU and torch sees none (try --layout cpu)")
    env: dict[str, str] = {}
    if gpu == "rocm" and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        if v := hsa_override_for(hw.get("name", "")):
            env["HSA_OVERRIDE_GFX_VERSION"] = v
    gpu_dev = gpu if gpu != "cpu" else "cpu"
    by = batch_by_role or {}

    def mb_for(rs):  # explicit flag > calibrated R > default
        return max_batch or max((by.get(r, 0) for r in rs), default=0) or None

    port = base_port + 1
    if layout == "together":
        return layout, [Backend(roles, gpu_dev, port, mb_for(roles) or 16, None, env)]
    if layout == "split-gpu":  # the decision model on the CPU, the narrator on the GPU
        out = []
        cpu_threads = max(1, cores - 2) if "s2" in roles else cores
        for r in roles:
            if r == "s1":
                out.append(Backend(("s1",), "cpu", port, mb_for(("s1",)) or 8, cpu_threads, {}))
            else:
                out.append(Backend(("s2",), gpu_dev, port, mb_for(("s2",)) or 8, None, env))
            port += 1
        return layout, out
    # cpu: one process per role, the cores shared
    share = max(1, cores // len(roles))
    out = []
    for r in roles:
        out.append(Backend((r,), "cpu", port, mb_for((r,)) or 8, share, {}))
        port += 1
    return "cpu", out


def detect_hardware() -> dict:
    env = dict(os.environ)
    first = subprocess.run(
        [sys.executable, "-m", "tle.liquid.hw"], capture_output=True, text=True, env=env, timeout=300
    )
    try:
        hw = json.loads(first.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"backend": "cpu", "name": "", "cores": os.cpu_count() or 1, "error": first.stderr[-300:]}
    return hw


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; it only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def make_link(host: str, port: int, key: str | None) -> str:
    return f"http://{host}:{port}" + (f"/k/{key}" if key else "")


def exports(link: str, roles: tuple[str, ...]) -> list[str]:
    if set(roles) == {"s1", "s2"}:
        return [f"export TLE_LIQUID_URL={link}"]
    return [f"export TLE_LIQUID_{r.upper()}_URL={link}" for r in roles]


def make_gateway(routes: dict[str, str], key: str | None, client=None):
    """Starlette app: ``routes`` maps a role ("s1"/"s2") to the backend base URL. The key may come as ``/k/<key>`` prefix or
    as ``Authorization: Bearer``."""
    import hmac

    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    client = client or httpx.AsyncClient(timeout=None)
    prefix = f"/k/{key}" if key else ""

    def role_of(path: str) -> str:
        return "s1" if path.endswith("/decisions") else "s2"

    async def handle(request: Request):
        path = request.url.path
        if key:
            if path.startswith(prefix + "/") or path == prefix:
                path = path[len(prefix) :] or "/"
            else:
                got = request.headers.get("authorization", "")
                if not hmac.compare_digest(got.encode(), f"Bearer {key}".encode()):
                    return JSONResponse({"error": {"message": "missing or wrong key"}}, status_code=401)
        if path == "/health":
            outs = {}
            for r, base in routes.items():
                try:
                    outs[r] = (await client.get(base + "/health")).json()
                except Exception as e:
                    outs[r] = {"status": f"down: {type(e).__name__}"}
            ok = all(v.get("status") == "ok" for v in outs.values())
            return JSONResponse({"status": "ok" if ok else "degraded", "backends": outs})
        role = role_of(path)
        if role not in routes:
            return JSONResponse(
                {"error": {"message": f"this machine does not serve {role} (serves {sorted(routes)})"}},
                status_code=404,
            )
        body = await request.body()
        deadline = time.monotonic() + RESTART_GRACE
        while True:
            try:
                r = await client.request(
                    request.method,
                    routes[role] + path,
                    content=body,
                    headers={"content-type": "application/json"},
                )
                break
            except (
                httpx.HTTPError
            ) as e:  # a backend that crashed (GPU hang) is restarted by the supervisor: wait for it
                if time.monotonic() > deadline:
                    return JSONResponse(
                        {"error": {"message": f"backend down: {type(e).__name__}"}}, status_code=502
                    )
                await asyncio.sleep(2)
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    return Starlette(routes=[Route("/{path:path}", handle, methods=["GET", "POST"])])


def _wait_health(url: str, procs: list[subprocess.Popen], timeout: float) -> None:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError(f"a server process exited with code {p.returncode} (see its log above)")
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"{url} did not become healthy in {timeout:.0f}s")


def supervise(backends, procs, envs, stop_flag):
    """Restart a model process that died (a GPU hang kills it): a long game must not end because of one crash."""
    restarts = 0
    while not stop_flag.is_set():
        for i, p in enumerate(procs):
            if p.poll() is not None and not stop_flag.is_set():
                restarts += 1
                print(
                    f"[up] {'+'.join(backends[i].roles)} exited with code {p.returncode}; restarting (#{restarts})",
                    flush=True,
                )
                time.sleep(5)  # let the driver recover
                procs[i] = subprocess.Popen(backends[i].argv(), env=envs[i])
        stop_flag.wait(2)


def run(
    roles: tuple[str, ...],
    port: int = 8100,
    share: bool = False,
    layout: str = "auto",
    max_batch: int | None = None,
    key: str | None = None,
    wait: float = 3600.0,
    hw: dict | None = None,
    dry_run: bool = False,
) -> int:
    import uvicorn

    from . import calibrate

    hw = hw or detect_hardware()
    cal = calibrate.load()
    by: dict[str, int] = {}
    if cal and calibrate.matches(cal, hw):
        by = {r: v["R"] for r, v in cal["roles"].items()}
        if layout == "auto":
            layout = cal["layout"]
        print(
            f"calibration: {cal['created']} (R per role {by}, G for S1 {cal['roles'].get('s1', {}).get('G', '-')})",
            flush=True,
        )
    else:
        cal = None
        print("calibration: none for this machine, using defaults (run `tle-liquid calibrate`)", flush=True)
    name, backends = plan(hw, roles, port, layout, max_batch, by)
    print(
        f"hardware: {hw.get('backend')} {hw.get('name', '')} ({hw.get('cores')} cores); layout: {name}",
        flush=True,
    )
    for b in backends:
        print(
            f"  {'+'.join(b.roles)} on {b.device} (port {b.port}, max_batch {b.max_batch}"
            + (f", {b.threads} threads" if b.threads else "")
            + ")",
            flush=True,
        )
    if dry_run:
        return 0
    key = (key or os.environ.get("TLE_LIQUID_API_KEY") or secrets.token_urlsafe(16)) if share else None
    procs: list[subprocess.Popen] = []
    envs: list[dict] = []
    stopping = threading.Event()

    def stop(*_):
        stopping.set()
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGTERM, lambda *a: (stop(), sys.exit(0)))
    routes: dict[str, str] = {}
    try:
        for b in backends:
            env = {**os.environ, **b.env}
            if b.threads:
                env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(b.threads)
            envs.append(env)
            procs.append(subprocess.Popen(b.argv(), env=env))
            for r in b.roles:
                routes[r] = f"http://127.0.0.1:{b.port}"
        for b in backends:
            _wait_health(f"http://127.0.0.1:{b.port}", procs, wait)
        host = lan_ip() if share else "127.0.0.1"
        link = make_link(host, port, key)
        print("\nREADY. On the machine that plays:\n", flush=True)
        for line in exports(link, roles):
            print("   ", line)
        print(
            f"\nlink: {link}\n(keep Ctrl-C to stop; {'the link contains the access key' if key else 'local only'})\n",
            flush=True,
        )
        threading.Thread(target=supervise, args=(backends, procs, envs, stopping), daemon=True).start()
        app = make_gateway(routes, key)
        uvicorn.run(app, host="0.0.0.0" if share else "127.0.0.1", port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print("error:", e, file=sys.stderr)
        return 1
    finally:
        stop()
    return 0
