import json

from tle.analysis import summary
from tle.balance import band_report, load_bands
from tle.contracts import models as m
from tle.core.rng import GameRng
from tle.experiments import Campaign, expand, run_campaign, run_one
from tle.loop.game import play
from tle.loop.strategies import STRATEGIES
from tle.models import CachedEngine, MockBackend, OpenAICompatBackend, RealEngine
from tle.runs import list_runs, table
from tle.telemetry import RunStore, TelemetryObserver
from tle.testing import catalogs, small_game

GOOD = json.dumps(
    {
        "pillar": "press",
        "domain": "press",
        "aggressiveness": "high",
        "predicate": {
            k: "any"
            for k in (
                "age_band",
                "social_class",
                "trade",
                "circle",
                "faction",
                "influencer",
                "ideology_extreme",
                "role",
            )
        },
        "tec_effect": "no effect",
        "cul_effect": "no effect",
        "econ_effect": "no effect",
        "threat": "none",
        "transfer": None,
        "revocation": None,
    }
)


def _eng(script, **kw):
    b = MockBackend(script)
    return b, RealEngine(b, catalogs().vocab(), **kw)


def test_real_engine_retries_then_succeeds():
    b, e = _eng({"classify_decree": ["not json", GOOD]})
    out = e.call("classify_decree", m.ClassifyDecreeIn(text="Close the gazette"), GameRng(1))
    rec = e.drain_execution()[0]
    assert out.pillar == "press" and rec.attempts == 2 and rec.fallbacks == 0


def test_real_engine_rejects_out_of_vocabulary_and_falls_back():
    bad = GOOD.replace('"press", "domain"', '"nonsense", "domain"')
    _, e = _eng({"classify_decree": [bad, bad, bad]}, n_retry=2)
    assert e.call("classify_decree", m.ClassifyDecreeIn(text="x"), GameRng(1)) is None
    assert e.drain_warnings() and e.drain_execution()[0].fallbacks == 1


def test_openai_compat_uses_injected_transport_and_strict_schema():
    seen = {}

    def transport(url, headers, body, timeout):
        seen["url"], seen["payload"] = url, json.loads(body)
        return json.dumps(
            {
                "choices": [{"message": {"content": GOOD}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        ).encode()

    be = OpenAICompatBackend(base_url="http://x/v1", model="m", transport=transport)
    e = RealEngine(be, catalogs().vocab())
    out = e.call("classify_decree", m.ClassifyDecreeIn(text="Close the gazette"), GameRng(1))
    assert out.pillar == "press" and seen["url"].endswith("/chat/completions")
    rf = seen["payload"]["response_format"]["json_schema"]
    assert rf["strict"] is True and rf["schema"]["additionalProperties"] is False
    assert e.drain_execution()[0].out_tokens == 7


def test_cache_replays_without_calling_the_model(tmp_path):
    b, e = _eng({"classify_decree": [GOOD]})
    c = CachedEngine(e, tmp_path / "c.jsonl")
    inp = m.ClassifyDecreeIn(text="Close the gazette")
    a = c.call("classify_decree", inp, GameRng(1))
    c2 = CachedEngine(e, tmp_path / "c.jsonl")  # a new process reading the same file
    b2 = c2.call("classify_decree", inp, GameRng(2))
    assert a == b2 and len(b.calls) == 1 and c2.hits == 1


def test_store_roundtrip_and_telemetry_is_a_pure_observer(tmp_path):
    g1 = small_game(2)
    store = RunStore(tmp_path, "r1", {"scenario": "x", "seed": 2}, {"a": 1}, flush_every=5)
    play(g1, STRATEGIES["random"], 20, [TelemetryObserver(store)])
    store.close()
    g2 = small_game(2)
    play(g2, STRATEGIES["random"], 20)
    assert g1.w.digest() == g2.w.digest()  # telemetry off is bit-identical
    metrics = table("r1", "metrics", tmp_path)
    assert len(metrics) == 20 and "approval" in metrics.columns
    assert len(table("r1", "events", tmp_path)) > 0
    assert list_runs(tmp_path)[0]["state"] == "finished"


def test_open_run_is_readable_before_close(tmp_path):
    g = small_game(2)
    store = RunStore(tmp_path, "open", {"scenario": "x", "seed": 2}, {}, flush_every=3)
    play(g, STRATEGIES["gradual"], 7, [TelemetryObserver(store)])
    assert len(table("open", "metrics", tmp_path)) >= 6  # streamed batches are readable (crash safety)


def test_campaign_designs_expand():
    base = dict(
        name="t", scenarios=["industrial_secular"], strategies=["gradual", "assault"], seeds=[1, 2], n=60
    )
    assert len(expand(Campaign(**base))) == 4
    assert (
        len(
            expand(
                Campaign(**base, design="grid", grid={"alpha_aware": [0.2, 0.4], "fear_decree": [0.1, 0.3]})
            )
        )
        == 16
    )
    oat = expand(Campaign(**base, design="oat", params=["alpha_react", "alpha_aware"], steps=3))
    assert len(oat) == 4 * (1 + 6)
    lhs = expand(Campaign(**base, design="lhs", params=["alpha_react"], samples=5))
    assert len(lhs) == 20


def test_run_one_is_deterministic_and_campaign_summarises(tmp_path):
    spec = {
        "campaign": "t",
        "scenario": "industrial_secular",
        "strategy": "gradual",
        "seed": 1,
        "n": 60,
        "overrides": {},
        "point": 0,
        "max_turns": None,
        "engine": "simulated",
    }
    assert run_one(spec)["digest"] == run_one(spec)["digest"]
    c = Campaign(
        name="t", scenarios=["industrial_secular"], strategies=["gradual", "assault"], seeds=[1, 2], n=60
    )
    df = run_campaign(c, tmp_path, workers=1, telemetry=False)
    assert len(df) == 4 and set(summary(df)["strategy"]) == {"gradual", "assault"}
    assert len(band_report(df, load_bands())) >= 1


def test_engines_are_configured_separately(tmp_path):
    from tle.models import build_engines, load_specs, with_overrides
    from tle.params import default_params

    f = tmp_path / "engines.toml"
    f.write_text(
        '[s1]\nkind = "openai_compat"\nbase_url = "http://jev:9/v1"\nmodel = "jev-x"\nstructured = false\n'
    )
    specs = with_overrides(load_specs(f), s2_kind="openai_compat", s2_model="narrator")
    eng, info = build_engines(catalogs(), default_params(), specs)
    assert info["s1"] == {"kind": "openai_compat", "model": "jev-x", "base_url": "http://jev:9/v1"}
    assert info["s2"]["model"] == "narrator" and eng.s1.backend.structured is False
    assert eng.s1.backend.base_url != eng.s2.backend.base_url or eng.s1.backend.model != eng.s2.backend.model
    _, sim = build_engines(catalogs(), default_params(), load_specs(None))
    assert sim == {"s1": {"kind": "simulated"}, "s2": {"kind": "simulated"}}


def test_jev_backend_turns_contracts_into_questions():
    from tle.models.jev import JevBackend

    seen = []

    def transport(url, headers, body, timeout):
        req = json.loads(body)
        seen.append((url, headers, req))
        ans = {}
        for k, q in req["questions"].items():
            first = next(iter(q["criteria"]))
            choice = "1" if k == "option" else ("press" if k == "pillar" else first)
            ans[k] = {"type": "choice", "choice": choice, "confidence": 0.9, "probabilities": {choice: 1.0}}
        return json.dumps({"answers": ans, "usage": {"input_tokens": 3, "output_tokens": 1}}).encode()

    be = JevBackend(api_key="k", transport=transport)
    e = RealEngine(be, catalogs().vocab(), fallback=None)
    # per-citizen choice among the options S2 wrote
    opts = [m.OptionIn(id=str(i), text=f"option {i}") for i in range(3)]
    news = m.NewsItem(
        event_id=1,
        title="t",
        text="x",
        event_type="crime",
        intensity="low",
        actor_kind="exogenous",
        target="none",
        tags=[],
    )
    res = e.call(
        "choose_reaction",
        m.ChooseReactionIn(context={"age_band": "adult"}, news=news, options=opts),
        GameRng(1),
    )
    assert res.option == "1"
    url, headers, req = seen[0]
    assert url.endswith("/decisions") and headers["Authorization"] == "Bearer k"
    assert req["questions"]["option"]["criteria"] == {"0": "option 0", "1": "option 1", "2": "option 2"}
    # a closed-vocabulary contract: one choice question per enum field, optional structures stay null
    out = e.call("classify_decree", m.ClassifyDecreeIn(text="Close the gazette"), GameRng(1))
    assert out is not None and out.pillar == "press" and out.transfer is None
    # a task Jev cannot express (a list) is a permanent failure: the engine falls back instead of retrying
    import pytest
    from tle.models.jev import JevUnsupported

    with pytest.raises(JevUnsupported):
        be.build("TASK: rank_event_types\nd\n\nINPUT:\n{}\n", e.schema("rank_event_types"))


def test_engines_file_expands_environment_variables(tmp_path, monkeypatch):
    from tle.models import load_dotenv, load_specs

    (tmp_path / ".env").write_text("TLE_X_MODEL=from-env\n# c\n")
    monkeypatch.delenv("TLE_X_MODEL", raising=False)
    load_dotenv(tmp_path / ".env")
    f = tmp_path / "e.toml"
    f.write_text(
        '[s1]\nkind = "jev"\nmodel = "${TLE_X_MODEL:-fallback}"\nbase_url = "${TLE_X_URL:-http://u}"\n'
    )
    s = load_specs(f)["s1"]
    assert s.model == "from-env" and s.base_url == "http://u"
