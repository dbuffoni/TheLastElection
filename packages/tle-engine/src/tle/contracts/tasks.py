"""Registry of the closed list of tasks (spec 13): each has a contract and an engine (S1 or S2)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from . import models as m


@dataclass(frozen=True)
class TaskSpec:
    name: str
    engine: str  # "s1" | "s2"
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    description: str


def _t(name, engine, i, o, desc):
    return TaskSpec(name, engine, i, o, desc)


TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in [
        _t(
            "citizen_profile",
            "s1",
            m.CitizenProfileIn,
            m.CitizenProfileOut,
            "Classify a description on temperament, character, ideology and trade",
        ),
        _t(
            "classify_decree",
            "s1",
            m.ClassifyDecreeIn,
            m.ClassifyDecreeOut,
            "Classify a decree text (<= 50 characters) onto the catalogs",
        ),
        _t(
            "rank_event_types",
            "s1",
            m.RankEventTypesIn,
            m.RankEventTypesOut,
            "Rank admitted event types for an actor",
        ),
        _t(
            "news_feature",
            "s1",
            m.NewsFeatureIn,
            m.NewsFeatureOut,
            "Reclassify the intensity of a news item and extract a transfer",
        ),
        _t(
            "option_feature",
            "s1",
            m.OptionFeatureIn,
            m.OptionFeatureOut,
            "Extract targets, strength and stances from a reaction option",
        ),
        _t(
            "choose_reaction",
            "s1",
            m.ChooseReactionIn,
            m.ChooseReactionOut,
            "Choose a reaction for a citizen",
        ),
        _t(
            "choose_activity",
            "s1",
            m.ChooseActivityIn,
            m.ChooseActivityOut,
            "Choose a daily activity for a citizen",
        ),
        _t(
            "pillar_level",
            "s1",
            m.PillarLevelIn,
            m.PillarLevelOut,
            "Propose the pillar level that a news item represents",
        ),
        _t(
            "choose_influencer_response",
            "s1",
            m.ChooseInfluencerResponseIn,
            m.ChooseInfluencerResponseOut,
            "Choose an influencer's response to a private message",
        ),
        _t("generate_citizens", "s2", m.GenerateCitizensIn, m.GenerateCitizensOut, "Describe new citizens"),
        _t(
            "narrate_event",
            "s2",
            m.NarrateEventIn,
            m.NarrateEventOut,
            "Write a news item and its reaction options from a seed",
        ),
        _t("name_faction", "s2", m.NameFactionIn, m.NameFactionOut, "Name a new faction"),
        _t(
            "message_responses",
            "s2",
            m.MessageResponsesIn,
            m.MessageResponsesOut,
            "Possible responses to a private message",
        ),
        _t(
            "newspaper_articles",
            "s2",
            m.NewspaperArticlesIn,
            m.NewspaperArticlesOut,
            "Write articles for the selected news items",
        ),
        _t("annual_edition", "s2", m.TextIn, m.TextOut, "Annual edition from the determinant log"),
        _t("almanac", "s2", m.TextIn, m.TextOut, "Final almanac from the determinant log"),
        _t(
            "suggest_decree",
            "s2",
            m.SuggestDecreeIn,
            m.SuggestDecreeOut,
            "Suggest decrees (<= 50 characters each) from labels",
        ),
    ]
}
