# Running S1 / S2 on another machine (or on this one) with `tle-liquid up`

`tle-liquid up` looks at the machine it runs on, starts the Liquid models with the layout that suits the hardware, and prints
**one link**. The machine that plays uses that link; nothing else has to be configured there.

## On the machine that serves (a clone of this repo)

```bash
packages/tle-liquid/scripts/remote.sh            # installs on first run (right torch wheel, weights), then serves both models
# or, already installed:   uv run tle-liquid up --share
```

It prints something like:

```
hardware: cuda NVIDIA GeForce RTX 4090 (16 cores); layout: together
  s1+s2 on cuda (port 8101, max_batch 16)
READY. On the machine that plays:
    export TLE_LIQUID_URL=http://192.168.1.50:8100/k/Qm3...
```

`--share` makes it listen on the network. The link contains a random access key (`/k/<key>`): whoever has the link can use the
models, so treat it like a password (plain HTTP: use a trusted network or a VPN / SSH tunnel). Without `--share` the link is
`http://127.0.0.1:8100` and only this machine can reach it.

## On the machine that plays

```bash
export TLE_LIQUID_URL=<the link>          # S1 and S2 both there
uv run tle run --engines config/engines.liquid.toml --n 60
```

Each engine can live somewhere else; one URL per role wins over the common one:

| you want | set |
|---|---|
| S1 and S2 both on the remote machine | `TLE_LIQUID_URL=<link>` |
| S1 remote, S2 local | run `tle-liquid up --roles s2` here, `TLE_LIQUID_S1_URL=<remote link>`, `TLE_LIQUID_S2_URL=http://127.0.0.1:8100` |
| both local | `tle-liquid up` (no `--share`), `TLE_LIQUID_URL=http://127.0.0.1:8100` |

Serve only one model from a machine with `--roles s1` or `--roles s2` (it then prints the matching `TLE_LIQUID_S1_URL` / `_S2_URL`).
Remote and local engines run at the same time: every model is its own process, so a slow S1 on the CPU never blocks S2 on a GPU, and
the game sends many requests at once (`max_concurrency` in `config/engines.liquid.toml`).

## What it chooses on its own

| hardware | layout | why |
|---|---|---|
| NVIDIA GPU, or a discrete AMD card | `together`: one process, S1 and S2 on the GPU | **not tested yet** (no such machine so far) |
| AMD integrated GPU (name ends in "Graphics") | `split-gpu`: S1 on the **CPU**, S2 on the GPU | measured on a Radeon 780M: both models on the iGPU hang the GPU under game load; this split completes games |
| no GPU | `cpu`: one process per model, the cores shared | S1 and S2 run in parallel instead of taking turns |

Force a layout with `--layout together|split-gpu|cpu`, check what would happen with `--dry-run`. For AMD APUs that PyTorch does
not support natively (780M/760M/740M) it sets `HSA_OVERRIDE_GFX_VERSION=11.0.0` for the model processes (your own value wins).

## Batches (the only tuning knob left)

The game sends **one citizen per request** (`batch_size = 1` in `config/engines.liquid.toml`): grouping several citizens in one
request gained 30-60% at best on the GPU and large requests hang small GPUs. The server fuses concurrent requests into one model
call: `--max-batch N` (default 16 on a dedicated GPU, 8 elsewhere). Measured on the 780M, S1 requests of about 10 citizens (~10 kB of
state) or less were safe; on the CPU batching changes nothing. Raise `--max-batch` on a big GPU and measure with
`uv run tle-liquid smoke`.

## Requirements and pitfalls

* `install.sh` picks the torch wheel (CUDA / ROCm / CPU) and installs `torchvision` too (d1's code imports it).
* ROCm with Triton needs Python headers (`Python.h`): use a uv-managed Python (`uv venv --managed-python -p 3.12`) or install `python3-dev`.
* Linux: the user must be in the `render` group for `/dev/kfd` and `/dev/dri/renderD*`.
* First start downloads the weights (about 6 GB) and loading takes about 10 s; `up` prints the link only when both models answer.
* Ctrl-C stops everything.
