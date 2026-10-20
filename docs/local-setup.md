# Running with a real model on your PC

The game talks to models only through `tle.engines.protocol.Engine`. The shipped real engine (`RealEngine`) speaks the
OpenAI chat-completions protocol with strict JSON-schema structured output, so any local server works.

## Pick a server
| Server | Command (example) | Notes |
|---|---|---|
| vLLM (Linux/NVIDIA) | `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000` | best throughput, true schema-constrained decoding |
| llama.cpp | `llama-server -m model.gguf -c 8192 --port 8000` | CPU/GPU, GGUF; supports `json_schema` grammar |
| Ollama | `ollama serve` then base url `http://localhost:11434/v1` | easiest on Windows/Mac |
| LM Studio | enable the local server | set the model name shown in the UI |

## Smoke test
```bash
uv run tle run --engine openai_compat --base-url http://localhost:8000/v1 --model <name> \
    --strategy gradual --n 60 --cache runs/cache.jsonl
uv run tle runs show <run_id> --table execution     # calls / attempts / fallbacks / tokens / latency per task
```
`runs/<run_id>/warnings.jsonl` holds every malformed answer. If fallbacks exceed ~2% adjust
`packages/tle-models/src/tle/models/prompts.py` or use a larger model. Failed answers retry `n_retry` times (parameter) and then
fall back to the simulated engine's rule, so the game never stops.

## S1 and S2 on different models
S1 (the closed-vocabulary classifier, Jev or a Jev-compatible model) and S2 (the narrator) are configured separately.
Copy `config/engines.toml`, set `kind = "openai_compat"`, `base_url` and `model` per engine, then:
```bash
uv run tle run --engines config/engines.toml --strategy gradual --n 60 --cache runs/cache.jsonl
# or by flags, e.g. Jev on a local server and S2 still simulated
uv run tle run --s1-engine openai_compat --s1-base-url http://localhost:8000/v1 --s1-model <jev-name> --n 60
```
`kind = "jev"` speaks TypeSafe's decision format (`POST .../api/alpha/decisions`, models such as `typesafe/jev-1.14`): the
closed vocabularies of each S1 contract become `choice` questions and S2's options become the criteria of a per-citizen
question. Tasks it cannot express (lists, free text) fall back to the simulated rule. Keep S2 on an LLM.

`structured = false` in the file turns off strict `json_schema` for servers that do not support it (answers are still validated
and retried). `api_key_env` names an environment variable that holds a key. The run manifest records which model served each engine.

## OpenRouter for both engines
`cp .env.example .env`, put your key in `OPENROUTER_API_KEY` (the `.env` file is git-ignored and stays local) and pick the
models in `TLE_S1_MODEL` / `TLE_S2_MODEL`. Then `uv run tle run --engines config/engines.openrouter.toml --n 60 --cache runs/cache.jsonl`.
S1 uses the Jev decision endpoint, S2 the chat-completions endpoint. Values in the engines file may use `${VAR}` or `${VAR:-default}`.

## Many items per request
With `openai_compat` the S1 engine answers up to `batch_size` (default 10) independent items of the same task in one request:
citizen profiles, per-citizen reactions and daily activities, option features. What the items share (the news and the options,
the anchor) is stated once; the answer is `{"items": [...]}` in the same order, validated item by item. An item the batch gets
wrong is asked again alone, so it still gets the retries and the simulated fallback. Set `batch_size = 1` to turn it off.
Requests have a hard deadline (`timeout`, 60 s) and a length cap (`max_tokens`, 2048): some providers loop under
schema-constrained decoding of free text, so S2 runs with `structured = false` in `config/engines.openrouter.toml`.

## Cost model
Per turn the pipeline makes: 1 narration + K_opt(10) option extractions per event, plus one reaction choice per involved citizen
(capped by `B_S1` = 40, the rest copy the nearest sampled citizen) and one daily-life choice per active citizen. Start with
`--n 60` and the cache; raise `--n` once latency is known. Use the simulated engine for balancing and the real one to validate.

## Not implemented yet: in-process `local` backend
`make_backend("local")` raises on purpose. Implement `Backend.complete` with transformers + outlines on the GPU machine if you
prefer loading weights in-process; nothing else changes.

## Optional extras
`uv sync --extra search` adds Optuna and SALib for Sobol/Optuna searches (hooked up through the `calibrate` skill).
