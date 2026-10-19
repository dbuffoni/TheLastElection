"""Closed catalogs defined at design time (spec 2.3, 12): pillars and levels, event types, activities,
trades, scenarios. The engine receives them as objects; ``tle-config`` loads them from TOML."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

AGE_BANDS = ("minor", "young", "adult", "elderly")
CLASSES = ("low", "middle", "high")
IDEOLOGY_AXES = ("secular_religious", "market_statist", "progressive_conservative")
TEMPERAMENT = ("NS", "HA", "RD", "P")
CHARACTER = ("SD", "CO", "ST")
THREATS = ("motion", "judicial")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PillarLevel(_M):
    id: str
    name: str
    # structured requirement (15.3): event types that represent the level and the minimum intensity index (0..6)
    event_types: list[str]
    min_intensity: int


class Pillar(_M):
    id: str
    name: str
    protects: str
    custodians: list[str]  # trades, or "councillor"
    threat: str | None = None
    levels: list[PillarLevel]

    @property
    def L(self) -> int:
        return len(self.levels)


class EventType(_M):
    id: str
    actors: list[str]  # mayor | faction | influencer | class | custodian | exogenous | threat | system
    requires: list[str] = Field(
        default_factory=list
    )  # preconditions on labels, evaluated by the events system
    target: str = "none"  # mayor | pillar | actor | none
    pillar: str | None = None  # target pillar id; "custodian" means the custodian's own pillar
    attacks: int = 0  # +1 erodes the target pillar, -1 defends it, 0 neutral
    polarizing: bool = False
    can_found: bool = False
    hostility: float = 0.0  # -1 supportive .. +1 confrontational (used by the simulated ranking)
    direction: str = "actor"  # ideological direction of the push: actor | religious | secular | statist | market | progressive | conservative | none
    econ: float = 0.0  # signed economic effect (bipolar value)
    tec: float = 0.0
    cul: float = 0.0
    t_C: list[float] = Field(
        default_factory=lambda: [float("nan")] * 3
    )  # character design direction (NaN = masked)
    weight: float = 1.0
    category: str = ""  # exogenous categories: crime | disaster | discovery | cultural | crisis
    predicate_kinds: list[str] = Field(
        default_factory=list
    )  # how exogenous involvement is drawn: circle | class | trade | age_band | all
    domain: str = "institutions"
    published: bool = True


class Activity(_M):
    id: str
    t_I: list[float]  # design target on ideology, NaN = masked
    stance_mayor: float | None = None  # explicit political stance (protest) else masked
    tec: float = 0.0
    cul: float = 0.0
    social_weight: float = 1.0
    professional: bool = False  # work creates professional ties between colleagues


class Trade(_M):
    id: str
    weight: float = 1.0
    custodian: bool = False


class Scenario(_M):
    id: str
    name: str
    axis1_mean: float
    axis1_sd: float
    bimodal: bool = False
    wealth_mean: float
    wealth_sd: float = 0.2
    cohesion_degree: float  # initial affinity ties per citizen
    cohesion_intensity: float
    rivalries_per_citizen: float = 0.0
    tec: float
    cul: float
    q_high: float  # share of the upper class
    q_low: float  # share of the lower class
    age_shares: dict[str, float]  # band -> share
    trade_multipliers: dict[str, float] = Field(default_factory=dict)
    overlay: dict[str, float] = Field(default_factory=dict)  # parameter overrides (layer 2 of the registry)


class Domain(_M):
    id: str
    name: str


class Catalogs(_M):
    pillars: list[Pillar]
    event_types: list[EventType]
    activities: list[Activity]
    trades: list[Trade]
    domains: list[Domain]
    scenarios: list[Scenario]

    # lookups are rebuilt lazily and are not part of the schema
    def pillar_ids(self) -> list[str]:
        return [p.id for p in self.pillars]

    def pillar(self, pid: str) -> Pillar:
        for p in self.pillars:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def pillar_index(self, pid: str) -> int:
        return self.pillar_ids().index(pid)

    def event_type(self, eid: str) -> EventType:
        for e in self.event_types:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def trade_ids(self) -> list[str]:
        return [t.id for t in self.trades]

    def activity_ids(self) -> list[str]:
        return [a.id for a in self.activities]

    def scenario(self, sid: str) -> Scenario:
        for s in self.scenarios:
            if s.id == sid:
                return s
        raise KeyError(f"unknown scenario '{sid}'; known: {[s.id for s in self.scenarios]}")

    def vocab(self) -> dict[str, list[str]]:
        """Closed vocabularies used to export the contract schemas as strict JSON schemas."""
        return {
            "pillars": self.pillar_ids(),
            "pillars_or_none": ["none", *self.pillar_ids()],
            "domains": [d.id for d in self.domains],
            "trades": self.trade_ids(),
            "activities": self.activity_ids(),
            "event_types": [e.id for e in self.event_types],
            "age_bands": list(AGE_BANDS),
            "classes": list(CLASSES),
        }
