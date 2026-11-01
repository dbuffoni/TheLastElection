# tle-liquid: the local model engine

Runs the two small [Liquid AI](https://www.liquid.ai/models) models of the game on your own GPU (or CPU) and serves them over HTTP.
It is separate from the game: the package never imports `tle.*` game code, and the game talks to it only through its existing
engine settings (`config/engines.liquid.toml`).

| Role | Model (Hugging Face) | What it does | API it serves |
|---|---|---|---|
| **S1** | `LiquidAI/d1-2B` (decision model, ~3.9 GB bf16) | picks one option from a closed list, one forward pass, no text | `POST /api/alpha/decisions` (what the game's `kind = "jev"` speaks) |
| **S2** | `LiquidAI/LFM2.5-1.2B-Instruct` (non-thinking, ~2.4 GB bf16) | writes the news, options, descriptions | `POST /v1/chat/completions` (OpenAI-compatible, `json_schema` output enforced by grammar) |

Also: `GET /health`, `GET /v1/models`. Smaller or larger models of the same families can be swapped in `config/liquid.toml`
(for example `LiquidAI/d1-omni-600M`, `LiquidAI/LFM2.5-350M`, `LiquidAI/LFM2.5-2.6B`).

**Batching.** Concurrent requests are fused into one GPU call (`max_batch` 16, waiting at most 15 ms for company), and the S2
grammar can differ per row. The game already sends many independent requests and `batch_size` items per request, so both layers stack.

**Hardware.** NVIDIA (CUDA), AMD (ROCm) and CPU use the same code: PyTorch's ROCm build answers to the `cuda` device. Weights load
straight onto the GPU (`device_map`), so little host RAM is needed. A 12 GB card holds both models with room for batches.

## Run it on this machine
```bash
packages/tle-liquid/scripts/install.sh --fetch     # detects NVIDIA / AMD / CPU, installs the matching torch, downloads the models
packages/tle-liquid/scripts/serve.sh               # http://127.0.0.1:8100
uv run tle run --engines config/engines.liquid.toml --n 60 --cache runs/cache.jsonl
```
`uv run tle-liquid doctor` shows what it found. Memory tight? `serve.sh --roles s2` (one model) or `--lazy` (load on first use),
or smaller repos in `config/liquid.toml`.

## Run it on your desktop, play from another machine
On the desktop (Fedora + NVIDIA: install the proprietary driver; the CUDA runtime comes with the torch wheel):
```bash
git clone <this repo> && cd TheLastElection
packages/tle-liquid/scripts/install.sh --fetch
TLE_LIQUID_API_KEY=pick-a-secret packages/tle-liquid/scripts/serve.sh --host 0.0.0.0
sudo firewall-cmd --add-port=8100/tcp          # add --permanent to keep it
```
On the machine that runs the game:
```bash
export TLE_LIQUID_URL=http://<desktop-ip>:8100
uv run tle run --engines config/engines.liquid.toml --n 60 --cache runs/cache.jsonl
```
If the server has a key, add `api_key_env = "TLE_LIQUID_API_KEY"` to both tables of the engines file and export the same key.
Prefer an SSH tunnel (`ssh -L 8100:localhost:8100 desktop`) over exposing the port on an untrusted network. A background service
file is in `scripts/tle-liquid.service`.

## Test it
```bash
uv run tle-liquid smoke --mode mock --game    # no model at all: HTTP contract, batching, 3 game turns through the server
uv run tle-liquid smoke --mode tiny --game    # random-weight LFM2 on the real S2 code path (CPU is enough), mock S1
uv run tle-liquid smoke                       # the real weights (after `fetch`)
```
`smoke` prints PASS/FAIL per check, including whether requests were really fused into batches.

## Limits you should know
* Verified on real weights (AMD Radeon 780M iGPU, ROCm 6.4 wheel, Linux): `tle-liquid smoke` passes 5/5 (S1 32 concurrent requests in
  batches of 16: 5 s on GPU vs 20 s on CPU; S2 constrained 127 tok/s vs 27). CUDA is untested.
* AMD iGPU (gfx1103): export `HSA_OVERRIDE_GFX_VERSION=11.0.0`, otherwise every kernel fails with `invalid device function`. Triton
  compiles small helpers at run time, so the Python in the venv needs its headers (`Python.h`): use a uv-managed Python
  (`uv venv --managed-python -p 3.12`) or install `python3-dev`. The user must be in the `render` group.
* **Open problem:** `smoke --game` is not green on that iGPU. The real game load (S2 constrained+sampled batches and S1 batches with
  ~110 questions each, with both models resident) hangs the GPU ("HW Exception ... GPU Hang") and kills the server, intermittently and
  not on the same request; the same requests replayed alone in a separate process succeed. A discrete GPU, or CPU, is not affected as far as tested.
* d1 ships its own code (`trust_remote_code`, needs `torchvision`). `system_one_batch` takes a list of `(state, questions)` tuples.
* The ROCm wheel index (`rocm6.4` in `install.sh`) may be old: set `TORCH_BACKEND` to the one your driver supports.
* One seed per fused batch: sampling with a `seed` is repeatable only for the same batch composition. S1 is deterministic.
