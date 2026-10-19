"""Frozen Pydantic schemas of the S1 and S2 tasks (spec 13).

Rules (Technical Implementation 3.7): every field is required (no defaults), closed vocabularies are
enums, no numeric bounds or formats in the exported schema; Pydantic-level checks run after parsing.
Fields marked ``vocab(...)`` take their enum from the catalogs when the schema is exported.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .vocab import Bipolar, BipolarOrMask, StanceOrMask, Unipolar


def vocab(name: str) -> Any:
    """Marks a string field whose values come from a catalog vocabulary (resolved at export/validate time)."""
    return Field(json_schema_extra={"x-vocab": name})


def vocab_list(name: str) -> Any:
    return Field(json_schema_extra={"x-vocab-items": name})


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- shared pieces -----------------------------------------------------------------------
class Predicate(Contract):
    """Recipients of a decree or event: for each dimension a catalog value or 'any' (5.13)."""

    age_band: str = vocab("age_bands+any")
    social_class: str = vocab("classes+any")
    trade: str = vocab("trades+any")
    circle: str  # 'any' or a circle id as text
    faction: str  # 'any', 'member' or 'non-member'
    influencer: str  # 'any', 'influencer' or 'non-influencer'
    ideology_extreme: str  # 'any' or '<axis>:<high|low>' with axis in 1..3
    role: str  # 'any' or 'councillor'


class NewsItem(Contract):
    event_id: int
    title: str
    text: str
    event_type: str = vocab("event_types")
    intensity: Unipolar
    actor_kind: str
    target: str
    tags: list[str]


class OptionIn(Contract):
    id: str
    text: str


class TransferOut(Contract):
    payers: Predicate
    beneficiaries: Predicate
    intensity: Unipolar
    duration: str  # 'one_off' | 'persistent'


# ---- S1 tasks ----------------------------------------------------------------------------
class CitizenProfileIn(Contract):
    description: str
    anchor: str  # description of the average citizen: the anchor of 0


class CitizenProfileOut(Contract):
    NS: Bipolar
    HA: Bipolar
    RD: Bipolar
    P: Bipolar
    SD: Bipolar
    CO: Bipolar
    ST: Bipolar
    I1: Bipolar
    I2: Bipolar
    I3: Bipolar
    trade: str = vocab("trades")


class ClassifyDecreeIn(Contract):
    text: str
    _structure: Any = PrivateAttr(default=None)  # simulated player: an already structured decree (15.5)


class ClassifyDecreeOut(Contract):
    pillar: str = vocab("pillars_or_none")
    domain: str = vocab("domains")
    aggressiveness: Unipolar
    predicate: Predicate
    tec_effect: BipolarOrMask
    cul_effect: BipolarOrMask
    econ_effect: BipolarOrMask
    threat: str  # 'none' | 'motion' | 'judicial' (threat the decree responds to)
    transfer: TransferOut | None
    revocation: Predicate | None


class RankEventTypesIn(Contract):
    actor_profile: dict[str, str]
    admitted: list[str] = vocab_list("event_types")
    parent_text: str


class RankEventTypesOut(Contract):
    order: list[str] = vocab_list("event_types")


class NewsFeatureIn(Contract):
    news: NewsItem


class NewsFeatureOut(Contract):
    intensity: Unipolar
    transfer: TransferOut | None


class OptionFeatureIn(Contract):
    news: NewsItem
    option_text: str
    _annotation: Any = PrivateAttr(default=None)  # simulated engines only: the hidden structured annotation


class OptionFeatureOut(Contract):
    t_I1: BipolarOrMask
    t_I2: BipolarOrMask
    t_I3: BipolarOrMask
    t_C1: BipolarOrMask
    t_C2: BipolarOrMask
    t_C3: BipolarOrMask
    e_r: BipolarOrMask
    F: Unipolar
    stance_mayor: StanceOrMask
    stance_pillar_1: StanceOrMask
    stance_pillar_2: StanceOrMask
    stance_pillar_3: StanceOrMask
    stance_pillar_4: StanceOrMask
    stance_pillar_5: StanceOrMask
    stance_actor: StanceOrMask
    stance_target: StanceOrMask


class ChooseReactionIn(Contract):
    context: dict[str, Any]  # citizen context in labels (6.5)
    news: NewsItem
    options: list[OptionIn]


class ChooseReactionOut(Contract):
    option: str


class ChooseActivityIn(Contract):
    context: dict[str, Any]


class ChooseActivityOut(Contract):
    activity: str = vocab("activities")


class PillarLevelIn(Contract):
    news: NewsItem
    pillar: str = vocab("pillars")
    levels: list[dict[str, str]]
    aggregate_outcome: dict[str, str]
    current_level: str


class PillarLevelOut(Contract):
    level: str


class ChooseInfluencerResponseIn(Contract):
    context: dict[str, Any]
    message: str
    options: list[OptionIn]


class ChooseInfluencerResponseOut(Contract):
    option: str


# ---- S2 tasks ----------------------------------------------------------------------------
class CitizenDraft(Contract):
    name: str
    description: str


class GenerateCitizensIn(Contract):
    scenario: str
    number: int
    ages: list[int]
    circles: list[int]
    parents: list[dict[str, str]]


class GenerateCitizensOut(Contract):
    citizens: list[CitizenDraft]


class EventSeed(Contract):
    event_id: int
    day: int
    parent: int | None
    parent_context: str
    actor: dict[str, str]
    event_type: str = vocab("event_types")
    target: dict[str, str]
    intensity: Unipolar
    involved_group: str
    characters: list[int]
    can_found: bool
    n_options: int


class NarrateEventIn(Contract):
    seed: EventSeed
    _hidden: Any = PrivateAttr(
        default=None
    )  # simulated engines only: numeric design inputs (actor ideology, signs)


class OptionOut(Contract):
    text: str
    adherence: bool


class NarrateEventOut(Contract):
    title: str
    text: str
    tags: list[str]
    options: list[OptionOut]
    _annotation: Any = PrivateAttr(default=None)  # simulated engines only: per-option hidden features


class NameFactionIn(Contract):
    seed_labels: dict[str, str]
    news: NewsItem


class NameFactionOut(Contract):
    name: str
    label: str


class MessageResponsesIn(Contract):
    message: str
    influencer_profile: dict[str, str]


class MessageResponsesOut(Contract):
    options: list[str]


class Article(Contract):
    title: str
    text: str


class NewspaperArticlesIn(Contract):
    items: list[NewsItem]


class NewspaperArticlesOut(Contract):
    articles: list[Article]


class TextIn(Contract):
    log: list[dict[str, str]]


class TextOut(Contract):
    text: str


class SuggestDecreeIn(Contract):
    indicators: dict[str, str]
    pillars: dict[str, str]


class SuggestDecreeOut(Contract):
    decrees: list[str]
