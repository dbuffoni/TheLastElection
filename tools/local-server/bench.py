#!/usr/bin/env python3
"""Check and benchmark the two local model servers (S1 decider, S2 narrator) with mock data.

Standard library only, so it runs on the desktop with no install. Python 3.11+.

  python3 bench.py check                 # one request to every server in bench.toml
  python3 bench.py s1                    # S1 accuracy + latency + throughput, every [[s1]] entry
  python3 bench.py s2                    # S2 validity + latency + tokens/s, every [[s2]] entry
  python3 bench.py all --only kev-4b     # both, restricted to entries whose name matches
  python3 bench.py selftest              # run against built-in fake servers (no GPU needed)

S1 speaks the System One format (POST {state, model, questions} -> answers[key].choice), the one
tle.models.jev already sends. S2 speaks OpenAI chat completions with a strict JSON schema, the one
tle.models.backends.OpenAICompatBackend sends. Mock data: mock_data.json (decrees labelled by a real S1).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
TIMEOUT = 120.0

# --- S1 questions (closed vocabularies of ClassifyDecreeOut) ----------------------------------------------------
PILLARS = {
    "none": "The decree restricts no pillar of democracy.",
    "pluralism": "Restricts opposition, elections or political associations.",
    "law": "Restricts the rule of law: courts, judges, equality before the law.",
    "press": "Restricts the free press or free information.",
    "property": "Restricts property rights: seizure, confiscation, expropriation.",
    "civil": "Restricts civil liberties: assembly, speech, worship, movement.",
}
DOMAINS = {k: k for k in ("institutions", "economy", "security", "press", "worship", "demographics")}
AGGR = ("null", "very low", "low", "medium", "high", "very high", "maximum")
Q_HELP = {
    "pillar": "Which pillar of democracy does the decree restrict or attack? 'none' if it restricts none.",
    "domain": "Which area of public life does the decree belong to?",
    "aggressiveness": "How forceful is the decree, from 'null' (a polite request) to 'maximum' (an outright seizure of power)?",
}
CRITERIA = {"pillar": PILLARS, "domain": DOMAINS, "aggressiveness": {a: a for a in AGGR}}

# --- S2 contract (NarrateEventOut) -------------------------------------------------------------------------------
S2_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "text", "tags", "options"],
    "properties": {
        "title": {"type": "string"},
        "text": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "adherence"],
                "properties": {"text": {"type": "string"}, "adherence": {"type": "boolean"}},
            },
        },
    },
}
S2_SYSTEM = (
    "You write short news items for a small town simulation. Never use digits or numbers: write quantities in words. "
    "Reply only with JSON: title (max 10 words), text (2 or 3 sentences), tags (2 to 4 lowercase words), "
    "options (3 or 4 short reactions a citizen could choose; adherence is true when the reaction backs the mayor)."
)


def http_json(url: str, body: dict, headers: dict | None = None) -> tuple[dict, float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:  # noqa: S310 (user-configured endpoint)
        data = json.loads(r.read())
    return data, time.perf_counter() - t0


def headers_of(cfg: dict) -> dict:
    return {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}


# --- VRAM sampling -------------------------------------------------------------------------------------------------
def vram_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return max(int(x) for x in out.split())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class VramPeak:
    def __init__(self) -> None:
        self.peak = vram_mb()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            v = vram_mb()
            if v is not None and (self.peak is None or v > self.peak):
                self.peak = v

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def run_load(fn, jobs: list, concurrency: int) -> tuple[list, float]:
    """Run fn(job) for every job with the given concurrency; returns (results, wall seconds)."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        res = list(pool.map(fn, jobs))
    return res, time.perf_counter() - t0


# --- S1 -------------------------------------------------------------------------------------------------------------
def s1_questions(prefix: str = "", suffix: str = "") -> dict:
    return {
        f"{prefix}{f}": {"type": "choice", "instructions": Q_HELP[f] + suffix, "criteria": CRITERIA[f]}
        for f in ("pillar", "domain", "aggressiveness")
    }


def s1_request(cfg: dict, decrees: list[dict]) -> tuple[list[dict] | None, float, str]:
    """One request for 1..k decrees. Returns (predicted labels per decree or None, seconds, error)."""
    if len(decrees) == 1:
        body = {"model": cfg["model"], "state": decrees[0]["text"], "questions": s1_questions()}
    else:
        state = "Decrees, by index:\n" + "\n".join(f"{j}: {d['text']}" for j, d in enumerate(decrees))
        qs: dict = {}
        for j in range(len(decrees)):
            qs.update(s1_questions(f"i{j}:", f" Answer for the decree at index {j}."))
        body = {"model": cfg["model"], "state": state, "questions": qs}
    try:
        data, dt = http_json(cfg["url"], body, headers_of(cfg))
        ans = data["answers"]
        pre = [""] if len(decrees) == 1 else [f"i{j}:" for j in range(len(decrees))]
        return [{f: ans[p + f]["choice"] for f in ("pillar", "domain", "aggressiveness")} for p in pre], dt, ""
    except Exception as e:  # noqa: BLE001 (a benchmark reports every failure)
        return None, 0.0, f"{type(e).__name__}: {e}"[:120]


def s1_accuracy(decrees: list[dict], preds: list[dict | None]) -> dict:
    ok = {"pillar": 0, "domain": 0, "aggressiveness": 0, "aggr_pm1": 0}
    n = 0
    for d, p in zip(decrees, preds, strict=True):
        if p is None:
            continue
        n += 1
        for f in ("pillar", "domain", "aggressiveness"):
            ok[f] += p[f] == d[f]
        ok["aggr_pm1"] += (
            p["aggressiveness"] in AGGR and abs(AGGR.index(p["aggressiveness"]) - AGGR.index(d["aggressiveness"])) <= 1
        )
    return {k: (v / n if n else float("nan")) for k, v in ok.items()} | {"answered": n}


def bench_s1(cfg: dict, data: dict, levels: list[int], batch_items: int, rounds: int) -> dict:
    decrees = data["s1_decrees"]
    out: dict = {"name": cfg["name"], "kind": "s1", "model": cfg["model"]}
    s1_request(cfg, decrees[:1])  # warm-up
    with VramPeak() as vp:
        # accuracy: every decree alone, concurrency 4
        res, _ = run_load(lambda d: s1_request(cfg, [d]), decrees, 4)
        out["accuracy"] = s1_accuracy(decrees, [r[0][0] if r[0] else None for r in res])
        out["errors"] = sorted({r[2] for r in res if r[2]})[:3]
        # batched: k decrees per request
        if batch_items > 1:
            groups = [decrees[i : i + batch_items] for i in range(0, len(decrees) - batch_items + 1, batch_items)]
            res, _ = run_load(lambda g: s1_request(cfg, g), groups, 4)
            flat_d = [d for g, r in zip(groups, res, strict=True) for d in g]
            flat_p = [p for g, r in zip(groups, res, strict=True) for p in (r[0] or [None] * len(g))]
            out["batched_accuracy"] = s1_accuracy(flat_d, flat_p) | {"items_per_request": batch_items}
        # load: citizens-per-turn style bursts
        out["load"] = {}
        for c in levels:
            jobs = [decrees[i % len(decrees)] for i in range(max(c * rounds, 8))]
            res, wall = run_load(lambda d: s1_request(cfg, [d]), jobs, c)
            lat = [r[1] for r in res if r[0]]
            out["load"][c] = {
                "req_per_s": len(lat) / wall,
                "p50_s": pct(lat, 0.5),
                "p95_s": pct(lat, 0.95),
                "failed": len(res) - len(lat),
            }
    out["vram_peak_mb"] = vp.peak
    return out


# --- S2 -------------------------------------------------------------------------------------------------------------
def s2_request(cfg: dict, seed: dict) -> dict:
    user = (
        f"TASK: narrate_event\nSeed: {seed['seed']}. Event type: {seed['event_type']}. "
        f"Intensity: {seed['intensity']}. The mayor's stance in this event: {seed['actor_stance']}."
    )
    body = {
        "model": cfg["model"],
        "temperature": 0.7,
        "max_tokens": cfg.get("max_tokens", 400),
        "messages": [{"role": "system", "content": S2_SYSTEM}, {"role": "user", "content": user}],
    }
    if cfg.get("structured", True):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": S2_SCHEMA},
        }
    try:
        data, dt = http_json(cfg["base_url"].rstrip("/") + "/chat/completions", body, headers_of(cfg))
        text = data["choices"][0]["message"]["content"]
        toks = data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:120], "dt": 0.0, "tokens": 0}
    return {"dt": dt, "tokens": toks, "error": "", **s2_check(text)}


def s2_check(text: str) -> dict:
    """Valid JSON of the contract, no digits (S2 never gives numbers), length bounds."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # a thinking model would show here
    try:
        o = json.loads(text)
        assert isinstance(o["title"], str) and isinstance(o["text"], str)
        assert isinstance(o["tags"], list) and all(isinstance(t, str) for t in o["tags"])
        assert 2 <= len(o["options"]) <= 6
        assert all(isinstance(x["text"], str) and isinstance(x["adherence"], bool) for x in o["options"])
    except (ValueError, KeyError, TypeError, AssertionError):
        return {"ok": False, "valid": False, "no_digits": False, "bounded": False}
    blob = o["title"] + o["text"] + " ".join(x["text"] for x in o["options"])
    no_digits = not re.search(r"\d", blob)
    bounded = len(o["title"]) <= 100 and len(o["text"]) <= 600
    return {"ok": no_digits and bounded, "valid": True, "no_digits": no_digits, "bounded": bounded}


def bench_s2(cfg: dict, data: dict, levels: list[int], rounds: int) -> dict:
    seeds = data["s2_seeds"]
    out: dict = {"name": cfg["name"], "kind": "s2", "model": cfg["model"]}
    s2_request(cfg, seeds[0])  # warm-up
    with VramPeak() as vp:
        out["load"] = {}
        for c in levels:
            jobs = [seeds[i % len(seeds)] for i in range(max(c * rounds, 12))]
            res, wall = run_load(lambda s: s2_request(cfg, s), jobs, c)
            good = [r for r in res if not r["error"]]
            lat = [r["dt"] for r in good]
            out["load"][c] = {
                "req_per_s": len(good) / wall,
                "tokens_per_s": sum(r["tokens"] for r in good) / wall,
                "p50_s": pct(lat, 0.5),
                "p95_s": pct(lat, 0.95),
                "valid": sum(r.get("valid", False) for r in good) / max(len(res), 1),
                "ok": sum(r.get("ok", False) for r in good) / max(len(res), 1),
                "failed": len(res) - len(good),
            }
            if c == levels[0]:
                out["errors"] = sorted({r["error"] for r in res if r["error"]})[:3]
    out["vram_peak_mb"] = vp.peak
    return out


# --- reporting ------------------------------------------------------------------------------------------------------
def fmt(x, nd=2) -> str:
    return "-" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(results: list[dict]) -> str:
    lines: list[str] = []
    s1 = [r for r in results if r["kind"] == "s1"]
    s2 = [r for r in results if r["kind"] == "s2"]
    if s1:
        lines += ["## S1 (decider)", "", "| name | model | pillar | domain | aggr | aggr ±1 | batched pillar | VRAM peak MB |", "|---|---|---|---|---|---|---|---|"]
        for r in s1:
            a, b = r["accuracy"], r.get("batched_accuracy", {})
            lines.append(
                f"| {r['name']} | {r['model']} | {fmt(a['pillar'])} | {fmt(a['domain'])} | {fmt(a['aggressiveness'])} | "
                f"{fmt(a['aggr_pm1'])} | {fmt(b.get('pillar'))} | {fmt(r['vram_peak_mb'])} |"
            )
        lines += ["", "| name | concurrency | req/s | p50 s | p95 s | failed |", "|---|---|---|---|---|---|"]
        for r in s1:
            for c, m in r["load"].items():
                lines.append(f"| {r['name']} | {c} | {fmt(m['req_per_s'])} | {fmt(m['p50_s'], 3)} | {fmt(m['p95_s'], 3)} | {m['failed']} |")
        for r in s1:
            if r["errors"]:
                lines.append(f"\n{r['name']} errors: {r['errors']}")
        lines.append("")
    if s2:
        lines += ["## S2 (narrator)", "", "| name | model | concurrency | req/s | tokens/s | p50 s | p95 s | valid | valid and clean | failed | VRAM peak MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s2:
            for c, m in r["load"].items():
                lines.append(
                    f"| {r['name']} | {r['model']} | {c} | {fmt(m['req_per_s'])} | {fmt(m['tokens_per_s'], 1)} | {fmt(m['p50_s'], 2)} | "
                    f"{fmt(m['p95_s'], 2)} | {fmt(m['valid'])} | {fmt(m['ok'])} | {m['failed']} | {fmt(r['vram_peak_mb'])} |"
                )
        for r in s2:
            if r.get("errors"):
                lines.append(f"\n{r['name']} errors: {r['errors']}")
        lines.append("")
    lines.append("valid = JSON of the contract; clean = valid, no digits, bounded length. VRAM is the whole GPU (nvidia-smi), so it includes the other server.")
    return "\n".join(lines)


# --- check ----------------------------------------------------------------------------------------------------------
def check(cfgs: dict) -> int:
    bad = 0
    data = json.loads((HERE / "mock_data.json").read_text())
    print(f"VRAM now: {fmt(vram_mb())} MB")
    for c in cfgs["s1"]:
        p, dt, err = s1_request(c, data["s1_decrees"][:1])
        print(f"S1 {c['name']:<14} {'OK ' + fmt(dt, 3) + ' s ' + json.dumps(p[0]) if p else 'FAIL ' + err}")
        bad += p is None
    for c in cfgs["s2"]:
        r = s2_request(c, data["s2_seeds"][0])
        ok = not r["error"]
        print(f"S2 {c['name']:<14} {'OK ' + fmt(r['dt'], 2) + ' s, valid=' + str(r.get('valid')) + ', clean=' + str(r.get('ok')) if ok else 'FAIL ' + r['error']}")
        bad += not ok
    return bad


# --- selftest: fake servers -----------------------------------------------------------------------------------------
def start_fakes() -> dict:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    gold = {d["text"]: d for d in json.loads((HERE / "mock_data.json").read_text())["s1_decrees"]}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(0.02)
            if self.path.endswith("/systemone"):
                state = body["state"]
                texts = [ln.split(": ", 1)[1] for ln in state.splitlines()[1:]] if state.startswith("Decrees") else [state]
                ans = {}
                for key in body["questions"]:
                    pre, f = (key.split(":", 1) if ":" in key else ("", key))
                    t = texts[int(pre[1:])] if pre else texts[0]
                    ans[key] = {"choice": gold[t][f], "probabilities": {}}
                out = {"answers": ans, "usage": {"input_tokens": 1, "output_tokens": 1}}
            else:
                o = {"title": "Town debates budget", "text": "The council argued late. Some residents are angry.",
                     "tags": ["budget", "council"], "options": [{"text": "Support", "adherence": True}, {"text": "Protest", "adherence": False}, {"text": "Wait", "adherence": False}]}
                out = {"choices": [{"message": {"content": json.dumps(o)}}], "usage": {"completion_tokens": 40}}
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    return {
        "s1": [{"name": "fake-s1", "url": f"http://127.0.0.1:{port}/v1/systemone", "model": "fake"}],
        "s2": [{"name": "fake-s2", "base_url": f"http://127.0.0.1:{port}/v1", "model": "fake"}],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "s1", "s2", "all", "selftest"])
    ap.add_argument("--config", default=str(HERE / "bench.toml"))
    ap.add_argument("--only", default="", help="substring of the entry name")
    ap.add_argument("--levels", default="1,4,16", help="concurrency levels")
    ap.add_argument("--rounds", type=int, default=3, help="requests per level = levels x rounds (at least 8)")
    ap.add_argument("--batch-items", type=int, default=5, help="S1 decrees per request in the batched test (1 = off)")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    if a.cmd == "selftest":
        cfgs = start_fakes()
        a.cmd = "all"
    else:
        cfgs = tomllib.loads(Path(a.config).read_text())
        cfgs = {"s1": cfgs.get("s1", []), "s2": cfgs.get("s2", [])}
    for k in ("s1", "s2"):
        cfgs[k] = [c for c in cfgs[k] if a.only in c["name"]]
    if a.cmd == "check":
        return 1 if check(cfgs) else 0

    data = json.loads((HERE / "mock_data.json").read_text())
    levels = [int(x) for x in a.levels.split(",")]
    results = []
    if a.cmd in ("s1", "all"):
        for c in cfgs["s1"]:
            print(f"S1 {c['name']} ...", file=sys.stderr)
            results.append(bench_s1(c, data, levels, a.batch_items, a.rounds))
    if a.cmd in ("s2", "all"):
        for c in cfgs["s2"]:
            print(f"S2 {c['name']} ...", file=sys.stderr)
            results.append(bench_s2(c, data, levels, a.rounds))
    md = report(results)
    print(md)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.json").write_text(json.dumps(results, indent=1))
    (out / f"{stamp}.md").write_text(md)
    print(f"\nsaved {out / (stamp + '.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
