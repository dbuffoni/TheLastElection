---
name: real-model-setup
description: Connect the game to a real local or hosted LLM (vLLM, llama.cpp server, Ollama, LM Studio, OpenAI-compatible) and smoke-test it.
---

# Real model setup

1. Start any OpenAI-compatible server, e.g. `vllm serve <model> --port 8000` (supports `response_format: json_schema`), or
   `llama-server -m model.gguf --port 8000`, or Ollama's `/v1` endpoint. See `docs/local-setup.md`.
2. Smoke test: `uv run tle run --engine openai_compat --base-url http://localhost:8000/v1 --model <name> --n 60 --cache runs/cache.jsonl --strategy gradual`.
   One game makes thousands of calls; start with `--n 60`.
3. Check quality in telemetry: `uv run tle runs show <run> --table execution` (attempts vs calls = retry rate, fallbacks = malformed
   answers) and `runs/<run>/warnings.jsonl` (the actual bad answers). Fallbacks above ~2% mean the prompt or schema needs work
   (`packages/tle-models/src/tle/models/prompts.py`) or the model is too small.
4. `--cache` makes reruns cheap and repeatable: identical (task, input) never calls the model twice.
5. Servers without strict structured output: `OpenAICompatBackend(structured=False)`; validation + retries still protect the game.
