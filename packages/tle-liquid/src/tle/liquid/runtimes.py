"""The two model runtimes. Each one takes a LIST of requests and answers a list (this is the unit the micro-batcher
calls), so batching is the normal path, not an add-on.

* ``DecisionRuntime``  - Liquid d1 (decision model). One forward pass per question set, no tokens generated; the weights
  ship their own ``system_one`` / ``system_one_batch`` code (``trust_remote_code``).
* ``ChatRuntime``      - an LFM instruct model with schema-constrained decoding (xgrammar): every row of a batch can have
  its own JSON schema and the answer always parses. Without xgrammar it generates freely and the game's validator retries.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from .config import ModelSpec
from .devices import DeviceInfo, pick_dtype

log = logging.getLogger("tle.liquid")


def _load_kwargs(dev: DeviceInfo, dtype) -> dict:
    kw: dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True}
    if dev.backend != "cpu":
        kw["device_map"] = {"": dev.torch_device}  # weights go straight to the GPU: little host RAM is needed
    return kw


def _strict_json(text: str) -> str:
    """The grammar lets raw control characters (a newline) through inside strings; strict JSON parsers refuse them. Re-encode."""
    try:
        json.loads(text)
        return text
    except ValueError:
        try:
            return json.dumps(json.loads(text, strict=False), ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            return text


class DecisionRuntime:
    kind = "decision"

    def __init__(self, spec: ModelSpec, path: str, dev: DeviceInfo, dtype: str = "auto"):
        import torch
        from transformers import AutoModel

        self.spec, self.dev = spec, dev
        self._torch = torch
        dt = pick_dtype(dev, dtype)
        try:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=spec.trust_remote_code, **_load_kwargs(dev, dt)
            )
        except (ValueError, TypeError, NotImplementedError):  # remote code without device_map support
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=spec.trust_remote_code, dtype=dt
            ).to(dev.torch_device)
        self.model.eval()
        self._batch_ok = hasattr(self.model, "system_one_batch")
        self._lock = threading.Lock()

    def run_batch(self, reqs: list[dict]) -> list[Any]:
        """``reqs``: [{"state": ..., "questions": {...}}]. Answers: [{"answers": {...}, "usage": {...}}]."""
        with self._lock, self._torch.inference_mode():
            if self._batch_ok and len(reqs) > 1:
                try:
                    out = self.model.system_one_batch(
                        [(r["state"], r["questions"]) for r in reqs]  # d1 takes (state, questions) tuples
                    )
                    if len(out) == len(reqs):
                        return list(out)
                except Exception as e:
                    log.warning("system_one_batch failed (%s: %s); answering one by one", type(e).__name__, e)
                    self._batch_ok = False
            res: list[Any] = []
            for r in reqs:
                try:
                    res.append(self.model.system_one(r["state"], r["questions"]))
                except Exception as e:
                    res.append(e)
            return res


class ChatRuntime:
    kind = "chat"

    def __init__(
        self, spec: ModelSpec, path: str, dev: DeviceInfo, dtype: str = "auto", max_new_tokens: int = 1024
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec, self.dev, self.max_new_tokens = spec, dev, max_new_tokens
        self._torch = torch
        dt = pick_dtype(dev, dtype)
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, **_load_kwargs(dev, dt))
        if dev.backend == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self._xgr = self._compiler = None
        self._compiled: dict[str, Any] = {}
        try:
            import xgrammar as xgr

            info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.vocab_size)
            self._xgr, self._compiler = xgr, xgr.GrammarCompiler(info)
        except Exception as e:
            log.warning("xgrammar unavailable (%s): structured output is validated by the client only", e)
        self._lock = threading.Lock()

    @property
    def constrained(self) -> bool:
        return self._compiler is not None

    def _grammar(self, schema: dict):
        key = json.dumps(schema, sort_keys=True)
        if key not in self._compiled:
            self._compiled[key] = self._compiler.compile_json_schema(
                key, any_whitespace=False
            )  # compact JSON: no whitespace loops
        return self._compiled[key]

    def run_batch(self, reqs: list[dict]) -> list[Any]:
        """``reqs``: [{"messages", "schema"|None, "temperature", "max_tokens", "seed"}] -> [{"text", "finish", "in", "out"}].
        Requests are grouped by whether they are constrained, so one ``generate`` call serves the whole group."""
        out: list[Any] = [None] * len(reqs)
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(reqs):
            groups.setdefault(
                (r.get("schema") is not None and self.constrained, r.get("temperature", 0.0) > 0), []
            ).append(i)
        for (constrained, sample), idx in groups.items():
            try:
                res = self._generate([reqs[i] for i in idx], constrained, sample)
            except Exception as e:
                res = [e] * len(idx)
            for i, r in zip(idx, res):
                out[i] = r
        return out

    def _generate(self, reqs: list[dict], constrained: bool, sample: bool) -> list[dict]:
        torch = self._torch
        prompts = [
            self.tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in reqs
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.dev.torch_device
        )
        n_new = min(self.max_new_tokens, max(int(r.get("max_tokens") or self.max_new_tokens) for r in reqs))
        kw: dict[str, Any] = {
            "max_new_tokens": n_new,
            "do_sample": sample,
            "pad_token_id": self.tok.pad_token_id,
        }
        if sample:
            kw.update(temperature=max(r["temperature"] for r in reqs), top_k=50)
        if constrained:
            kw["logits_processor"] = [
                self._xgr.contrib.hf.LogitsProcessor([self._grammar(r["schema"]) for r in reqs])
            ]
        with self._lock, torch.inference_mode():
            seeds = [r["seed"] for r in reqs if r.get("seed") is not None]
            if seeds:
                torch.manual_seed(
                    int(seeds[0])
                )  # one generator per batch: row-level seeds do not exist in a fused call
            gen = self.model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1] :].cpu()
        results = []
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        for i, r in enumerate(reqs):
            row = new[i].tolist()
            stop_at = next(
                (k for k, t in enumerate(row) if t in eos), None
            )  # position of the first end-of-text token
            cap = int(r.get("max_tokens") or n_new)
            n_out = len(row) if stop_at is None else stop_at + 1
            finish = "stop" if stop_at is not None and n_out <= cap else "length"
            n_out = min(n_out, cap)
            text = self.tok.decode(row[:n_out], skip_special_tokens=True)
            if constrained and finish == "stop":
                text = _strict_json(text)
            results.append(
                {
                    "text": text,
                    "finish": finish,
                    "in": int(enc["attention_mask"][i].sum()),
                    "out": n_out,
                }
            )
        return results
