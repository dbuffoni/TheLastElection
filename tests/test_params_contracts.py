import pytest
from tle.contracts.schema import check_strict_subset, export_all
from tle.contracts.tasks import TASKS
from tle.params import default_params, from_flat, to_flat
from tle.registry import codegen
from tle.registry.loader import load_registry
from tle.testing import catalogs

pytestmark = pytest.mark.inc01


def test_generated_params_are_current():
    assert codegen.is_current(), "run `uv run tle params codegen`"


def test_every_registry_entry_is_in_params():
    flat = to_flat(default_params())
    missing = [k for k in load_registry().specs if k not in flat]
    assert not missing


def test_overrides_recompute_derived_and_reject_unknown():
    p = default_params()
    q = p.with_overrides({"N": 600})
    assert q.time.N == 600 and q.factions.N_min >= p.factions.N_min
    with pytest.raises(KeyError):
        p.with_overrides({"not_a_param": 1})
    with pytest.raises(KeyError):
        p.with_overrides({"N_min": 3})  # derived: not settable


def test_flat_roundtrip():
    p = default_params()
    assert to_flat(from_flat(to_flat(p))) == to_flat(p)


def test_all_schemas_are_strict_subset():
    for name, sch in export_all(catalogs().vocab()).items():
        assert check_strict_subset(sch) == [], name


def test_catalog_cross_references():
    cat = catalogs()
    pillars = set(cat.pillar_ids())
    for et in cat.event_types:
        if et.pillar not in (None, "custodian"):
            assert et.pillar in pillars, et.id
    assert len(cat.event_types) >= 40 and len(cat.pillars) == 5
    assert all(len(p.levels) == 6 for p in cat.pillars)
    for p in cat.pillars:
        assert [lv.min_intensity for lv in p.levels] == sorted(lv.min_intensity for lv in p.levels)


def test_every_task_has_a_simulated_handler():
    from tle.engines.simulated import SimulatedEngine

    e = SimulatedEngine(catalogs(), default_params())
    batched = {"choose_reaction": "choose_reaction_many", "choose_activity": "choose_activity_many"}
    for t in TASKS:
        assert hasattr(e, batched.get(t, f"_t_{t}")), t
