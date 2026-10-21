"""Serializable world state (spec 2, Technical Implementation 2).

Structure of arrays, one array per citizen attribute. No globals, no singletons. The state is
extended additively: every new field has a neutral default, nothing is renamed or removed; old
snapshots missing a field load with the default.
"""

from __future__ import annotations

import io
import json
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import numpy as np
from tle.core.digest import combine, hash_arrays
from tle.core.rng import GameRng

N_PILLARS = 5
N_CLASSES = 3
F_CAP = 64  # capacity of the faction table
BOND_TYPES = ("family", "affinity", "professional", "rivalry", "political")
B_FAM, B_AFF, B_PRO, B_RIV, B_POL = range(5)
KIN_NONE, KIN_PARENT, KIN_SIBLING, KIN_PARTNER = range(4)


def _z(shape, dtype=float):
    return field(default_factory=lambda: np.zeros(shape, dtype=dtype))


def _f(shape, dtype=float):
    return _z(shape, dtype)


@dataclass
class World:
    # ---- meta (JSON) ---------------------------------------------------------------------
    seed: int = 0
    scenario: str = ""
    n_cap: int = 0
    n_used: int = 0
    day: float = 0.0
    turn: int = 0
    dt_prev: float = 1.0
    next_event_id: int = 1
    next_fac_id: int = 1
    outcome: str = ""  # "" while playing; victory | removed | term_end
    outcome_day: float = -1.0
    warmup_done: bool = False
    rng_state: str = ""
    # ---- citizens ------------------------------------------------------------------------
    alive: np.ndarray = _f(0, bool)
    age: np.ndarray = _f(0)
    T: np.ndarray = _f((0, 4))
    C: np.ndarray = _f((0, 3))
    I: np.ndarray = _f((0, 3))
    B: np.ndarray = _f((0, 3))  # moving ideological base
    O: np.ndarray = _f((0, 6))  # private opinion: mayor + five pillars
    Ob: np.ndarray = _f((0, 6))  # base of the opinion (0, except councillors)
    r: np.ndarray = _f(0)
    r0: np.ndarray = _f(0)
    r_prev: np.ndarray = _f(0)  # wealth at the previous turn (economic vote)
    dr_transfer: np.ndarray = _f(0)  # wealth change of the turn due to transfers
    d_r_transfer: np.ndarray = _f(0)  # wealth change due to transfers in the last turn
    cls: np.ndarray = _f(0, np.int8)
    trade: np.ndarray = _f(0, np.int16)
    circle: np.ndarray = _f(0, np.int16)
    councillor: np.ndarray = _f(0, np.int8)  # 0 none, +1 majority side, -1 opposition side
    last_s1: np.ndarray = _f(0)
    lab_I: np.ndarray = _f((0, 3), np.int8)  # previous labels (hysteresis)
    lab_O: np.ndarray = _f((0, 6), np.int8)
    fear: np.ndarray = _f(0)
    arrival_day: np.ndarray = _f(0)
    parent1: np.ndarray = _f(0, np.int32)
    parent2: np.ndarray = _f(0, np.int32)
    couple: np.ndarray = _f(0, np.int32)  # partner id or -1
    names: list = field(default_factory=list)
    # ---- bonds (sparse) --------------------------------------------------------------------
    bn: int = 0
    bsrc: np.ndarray = _f(0, np.int32)
    bdst: np.ndarray = _f(0, np.int32)
    btype: np.ndarray = _f(0, np.int8)
    bkin: np.ndarray = _f(0, np.int8)
    bw: np.ndarray = _f(0)
    bstable: np.ndarray = _f(0, bool)
    blast: np.ndarray = _f(0)
    # ---- factions --------------------------------------------------------------------------
    fac_alive: np.ndarray = _f(F_CAP, bool)
    fac_id: np.ndarray = _f(F_CAP, np.int32)
    fac_mu: np.ndarray = _f((F_CAP, 3))
    fac_S: np.ndarray = _f(F_CAP)
    fac_birth: np.ndarray = _f(F_CAP)
    fac_last_action: np.ndarray = _f(F_CAP)
    fac_prev_S: np.ndarray = _f(F_CAP)
    fac_name: list = field(default_factory=list)
    fac_m: np.ndarray = _f((0, F_CAP))  # commitment per citizen and faction
    fac_member: np.ndarray = _f((0, F_CAP), bool)
    fac_last_active: np.ndarray = _f((0, F_CAP))
    # ---- influencers -----------------------------------------------------------------------
    inf_on: np.ndarray = _f(0, bool)
    inf_K: np.ndarray = _f(0)
    inf_since: np.ndarray = _f(0)
    inf_last_action: np.ndarray = _f(0)
    # ---- narrative pressures (classes, pillar custodians) ------------------------------------
    p_class: np.ndarray = _f(N_CLASSES)
    p_cust: np.ndarray = _f(N_PILLARS)
    # ---- power -----------------------------------------------------------------------------
    K: float = 0.0
    deficit_total: float = 0.0
    decree_success: float = 0.5
    pillar_level: np.ndarray = _f(N_PILLARS, np.int8)
    pillar_fall_day: np.ndarray = _f(N_PILLARS)
    threat_p: np.ndarray = _f(2)
    threat_phase: np.ndarray = _f(2, np.int8)  # 0 inactive, 1..3 phases
    threat_next: np.ndarray = _f(2)
    threat_events: list = field(default_factory=lambda: [[], []])
    threat_history: list = field(default_factory=list)
    # ---- community & economy -----------------------------------------------------------------
    tec: float = 0.0
    cul: float = 0.0
    tec_base: float = 0.0
    cul_base: float = 0.0
    extractiveness: float = 0.0
    coalition: np.ndarray = _f(0, bool)
    y_rent: np.ndarray = _f(0)  # rent received accumulator
    y_last_active: np.ndarray = _f(0)
    rents: list = field(default_factory=list)
    # ---- indicators (moving averages and previous labels) ---------------------------------------
    ind: dict = field(default_factory=dict)
    ind_lab: dict = field(default_factory=dict)
    # ---- narrative / logs ------------------------------------------------------------------------
    events: list = field(default_factory=list)  # recent events (older ones live in the telemetry stream)
    contrib: dict = field(default_factory=dict)  # accumulator key -> [(event id, amount, day)]
    memory: list = field(default_factory=list)  # determinant log (event id, day, title)
    pending_inputs: list = field(default_factory=list)
    last_player_day: float = -1e9
    stats: dict = field(default_factory=dict)  # counters used by metrics (jumps, falls, ...)

    # ---- helpers ----------------------------------------------------------------------------
    @property
    def rng(self) -> GameRng:
        return GameRng.from_state(self.rng_state)

    def save_rng(self, rng: GameRng) -> None:
        self.rng_state = rng.get_state()

    def adults(self, a_adult: float = 18.0) -> np.ndarray:
        return self.alive & (self.age >= a_adult)

    def ids(self, mask) -> np.ndarray:
        return np.flatnonzero(mask)

    # ---- serialization ------------------------------------------------------------------------
    @classmethod
    def _array_fields(cls) -> list[str]:
        out = []
        for f in fields(cls):
            d = f.default_factory() if f.default_factory is not MISSING else None
            if isinstance(d, np.ndarray):
                out.append(f.name)
        return out

    def to_bytes(self) -> bytes:
        arrays = {k: getattr(self, k) for k in self._array_fields()}
        meta = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in arrays}
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            __meta__=np.frombuffer(json.dumps(meta, default=_json_default).encode(), dtype=np.uint8),
            **arrays,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> World:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        meta = json.loads(bytes(z["__meta__"]).decode())
        w = cls()
        arrays = set(cls._array_fields())
        for f in fields(cls):
            if f.name in arrays:
                if f.name in z.files:
                    setattr(w, f.name, np.array(z[f.name]))  # missing fields keep the neutral default
            elif f.name in meta:
                setattr(w, f.name, meta[f.name])
        return w

    def copy(self) -> World:
        return World.from_bytes(self.to_bytes())

    # ---- digest (per state group) ----------------------------------------------------------------
    GROUPS = {
        "citizens": (
            "alive",
            "age",
            "T",
            "C",
            "I",
            "B",
            "O",
            "Ob",
            "r",
            "r0",
            "cls",
            "trade",
            "circle",
            "councillor",
            "last_s1",
            "couple",
        ),
        "bonds": ("bsrc", "bdst", "btype", "bkin", "bw", "bstable", "blast"),
        "factions": ("fac_alive", "fac_mu", "fac_S", "fac_m", "fac_member"),
        "influencers": ("inf_on", "inf_K"),
        "power": ("pillar_level", "threat_p", "threat_phase", "threat_next", "p_class", "p_cust"),
        "scalars": ("day", "turn", "K", "tec", "cul", "extractiveness", "decree_success", "outcome"),
        "rng": ("rng_state",),
    }

    def digest_groups(self) -> dict[str, str]:
        out = {}
        for g, names in self.GROUPS.items():
            items = []
            for n in names:
                v = getattr(self, n)
                if isinstance(v, np.ndarray) and n == "bsrc":
                    v = v[: self.bn]
                items.append((n, v))
            if g == "bonds":
                items = [(n, getattr(self, n)[: self.bn]) for n in names]
            out[g] = hash_arrays(items)
        return out

    def digest(self) -> str:
        return combine(self.digest_groups())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def allocate(w: World, cap: int) -> None:
    """Allocate every per-citizen array with ``cap`` slots (neutral defaults)."""
    w.n_cap = cap
    w.alive = np.zeros(cap, bool)
    w.age = np.zeros(cap)
    w.T = np.zeros((cap, 4))
    w.C = np.zeros((cap, 3))
    w.I = np.zeros((cap, 3))
    w.B = np.zeros((cap, 3))
    w.O = np.zeros((cap, 6))
    w.Ob = np.zeros((cap, 6))
    w.r = np.zeros(cap)
    w.r0 = np.zeros(cap)
    w.r_prev = np.zeros(cap)
    w.dr_transfer = np.zeros(cap)
    w.d_r_transfer = np.zeros(cap)
    w.cls = np.zeros(cap, np.int8)
    w.trade = np.zeros(cap, np.int16)
    w.circle = np.zeros(cap, np.int16)
    w.councillor = np.zeros(cap, np.int8)
    w.last_s1 = np.zeros(cap)
    w.lab_I = np.full((cap, 3), 3, np.int8)
    w.lab_O = np.full((cap, 6), 3, np.int8)
    w.fear = np.zeros(cap)
    w.arrival_day = np.zeros(cap)
    w.parent1 = np.full(cap, -1, np.int32)
    w.parent2 = np.full(cap, -1, np.int32)
    w.couple = np.full(cap, -1, np.int32)
    w.fac_m = np.zeros((cap, F_CAP))
    w.fac_member = np.zeros((cap, F_CAP), bool)
    w.fac_last_active = np.zeros((cap, F_CAP))
    w.inf_on = np.zeros(cap, bool)
    w.inf_K = np.zeros(cap)
    w.inf_since = np.zeros(cap)
    w.inf_last_action = np.full(cap, -1e9)
    w.coalition = np.zeros(cap, bool)
    w.y_rent = np.zeros(cap)
    w.y_last_active = np.zeros(cap)
    w.names = [""] * cap
    w.fac_id = np.zeros(F_CAP, np.int32)
    w.fac_name = [""] * F_CAP
    w.bsrc = np.zeros(1024, np.int32)
    w.bdst = np.zeros(1024, np.int32)
    w.btype = np.zeros(1024, np.int8)
    w.bkin = np.zeros(1024, np.int8)
    w.bw = np.zeros(1024)
    w.bstable = np.zeros(1024, bool)
    w.blast = np.zeros(1024)
