# Running S1 / S2 on another machine (or on this one) with `tle-liquid up`

`tle-liquid up` looks at the machine it runs on, starts the Liquid models with the layout that suits the hardware, and prints
**one link**. The machine that plays uses that link; nothing else has to be configured there.

## The three steps

| step | command | what it does |
|---|---|---|
| 1. configuration | `packages/tle-liquid/scripts/install.sh auto --fetch` | right torch wheel (CUDA / ROCm / CPU), libraries, weights |
| 2. calibration | `uv run tle-liquid calibrate` | finds the best **G** and **R** for this machine and saves them in `.tle-liquid/calibration.json` |
| 3. activation | `uv run tle-liquid up --share` | starts the models with that layout and batches, prints the link |

`scripts/remote.sh` does all three (skipping what is already done; `--recalibrate` forces step 2, `--quick` uses a small grid).

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
| AMD integrated GPU (name ends in "Graphics") | `s1-gpu`: S1 on the **GPU**, S2 on the CPU | S1 is the busy model (one call per citizen per event, ~80% of the calls) and the GPU makes it ~5x faster; S2 writes a few texts per turn. Both on the iGPU hung the GPU under game load (measured on a 780M), one per device is stable |
| no GPU | `cpu`: one process per model, the cores shared | S1 and S2 run in parallel instead of taking turns |

Force a layout with `--layout together|s1-gpu|s2-gpu|cpu`, check what would happen with `--dry-run`. For AMD APUs that PyTorch does
not support natively (780M/760M/740M) it sets `HSA_OVERRIDE_GFX_VERSION=11.0.0` for the model processes (your own value wins).

## G and R, and calibration

* **G** = citizens per request (S1 only). The game sends G citizens in one request. It is a client setting, `batch_size` in
  `config/engines.liquid.toml`, read from `TLE_LIQUID_S1_BATCH` (default 1 = no grouping).
* **R** = requests fused into one model call. It is a server setting, `--max-batch`. It only matters if the game sends that many
  requests at once (`max_concurrency` in the engines file, 16 by default).

`tle-liquid calibrate` measures them with **real requests of the game that are stored in the repo**
(`packages/tle-liquid/data/s1_requests.jsonl`, `s2_requests.jsonl`; regenerate with
`scripts/capture_calibration_data.py`), so a fresh clone needs no game run:

1. detects the hardware and uses the same layout / devices / CPU threads as `up`;
2. times a grid of (G, R) per role: S1 on the reaction and the activity requests (what runs every turn), S2 on its generation
   requests (G is not used there). Every configuration runs in a worker process: a size that hangs the GPU kills only that worker,
   and every configuration at least as big is skipped;
3. takes the fastest configuration (within 5% of the best, the smaller G and R win), runs it three more times and keeps it only if all
   three succeed, otherwise tries the next one;
4. saves `.tle-liquid/calibration.json` (git-ignored, one per machine, with the whole grid).

`up` reads that file when the hardware matches: it uses the calibrated layout and R, and prints
`export TLE_LIQUID_S1_BATCH=<G>` next to the link, to be set on the machine that plays. Without a calibration it uses safe defaults
(R = 16 on a dedicated GPU, 8 elsewhere; G = 1). `--max-batch` always wins over the file.

Grids: GPU `G in {1,2,5,10}`, `R in {1,2,4,8,16}` (at most 64 citizens per measurement); CPU `G in {1,5}`, `R in {1,2,4}`;
`--quick` is smaller. On the 780M the earlier hand measurements were: grouping above ~10-12 citizens hangs the GPU and gains 30-60% at
best; on the CPU neither G nor R changes anything.

## Similarity cache (optional, off by default)

`similarity` in `config/engines.liquid.toml` (env `TLE_LIQUID_SIMILARITY`, 0..1, -1 = off): before asking S1, citizens are compared on
their S7 labels (`packages/tle-models/src/tle/models/similarity.py`); a citizen closer than the threshold to one that was already
answered for the same question (same news and options) copies that answer, so the model sees only the representatives. Deterministic
(index order, nearest wins). Hits show in the execution table as `choose_reaction~similar` / `choose_activity~similar`.
Measured with the real d1 on 60 citizens (29 reaction and 20 activity calls): reactions saved 3% / 19% / 33% of the calls at
threshold 0.04 / 0.06 / 0.08 with followers matching their representative 76% / 75% / 70% of the time, a random citizen's answer
matching 65%; activities saved 1% / 8% / 9% with 100% / 89% / 90% agreement (random: 41%). So the label distance predicts the
activity well but the reaction poorly: the cache is a speed-for-fidelity trade, not free. Fitting per-label weights did not help.

## Requirements and pitfalls

* `install.sh` picks the torch wheel (CUDA / ROCm / CPU) and installs `torchvision` too (d1's code imports it).
* ROCm with Triton needs Python headers (`Python.h`): use a uv-managed Python (`uv venv --managed-python -p 3.12`) or install `python3-dev`.
* Linux: the user must be in the `render` group for `/dev/kfd` and `/dev/dri/renderD*`.
* First start downloads the weights (about 6 GB) and loading takes about 10 s; `up` prints the link only when both models answer.
* Ctrl-C stops everything.
