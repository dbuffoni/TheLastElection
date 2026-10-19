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

## Cost model
Per turn the pipeline makes: 1 narration + K_opt(10) option extractions per event, plus one reaction choice per involved citizen
(capped by `B_S1` = 40, the rest copy the nearest sampled citizen) and one daily-life choice per active citizen. Start with
`--n 60` and the cache; raise `--n` once latency is known. Use the simulated engine for balancing and the real one to validate.

## Not implemented yet: in-process `local` backend
`make_backend("local")` raises on purpose. Implement `Backend.complete` with transformers + outlines on the GPU machine if you
prefer loading weights in-process; nothing else changes.

## Optional extras
`uv sync --extra search` adds Optuna and SALib for Sobol/Optuna searches (hooked up through the `calibrate` skill).
