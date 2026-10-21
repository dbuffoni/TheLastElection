"""Prompt rendering. One system prompt per engine role and one user prompt per task; the user prompt
always starts with ``TASK: <name>`` and ends with the JSON input. Prompts are data (this file), not logic:
tune them here, never in the pipeline."""

from __future__ import annotations

import json

from pydantic import BaseModel
from tle.contracts.tasks import TASKS

SYSTEM = {
    "s1": (
        "You are the classifier of a political simulation. You never invent numbers. You answer ONLY with a JSON "
        "object that matches the given schema, choosing labels from the closed vocabularies. 'no effect' means "
        "the text says nothing about that feature."
    ),
    "s2": (
        "You are the narrator of a political simulation set in a small town. You write short, concrete, vivid text. "
        "You never mention numbers or scores. You answer ONLY with a JSON object that matches the given schema."
    ),
}


# Extra constraints per task (small models repeat themselves without a length bound).
NOTES = {
    "generate_citizens": (
        "Write exactly one entry per requested citizen, in order. Each description is ONE sentence of at most 25 words "
        "(trade, temperament, political leanings); never repeat phrases."
    ),
    "narrate_event": "Keep every text under 60 words and every option under 20 words; never repeat phrases.",
}


def render(task: str, inp: BaseModel) -> tuple[str, str]:
    spec = TASKS[task]
    note = f"\n{NOTES[task]}" if task in NOTES else ""
    dump = inp.model_dump()
    if isinstance(dump.get("context"), dict) and "temperament" in dump["context"]:
        dump["context"] = describe_citizen(dump["context"])  # words, not bare numbers
    user = f"TASK: {task}\n{spec.description}{note}\n\nINPUT:\n{json.dumps(dump, ensure_ascii=False)}\n"
    return SYSTEM[spec.engine], user


def render_batch(task: str, shared: dict, items: list[dict]) -> tuple[str, str]:
    """One request for k independent items of the same task: what they have in common is stated once."""
    spec = TASKS[task]
    k = len(items)
    user = (
        f"TASK: {task}\n{spec.description}\n\n"
        f"This is a BATCH of {k} independent items. Treat each item on its own, never let one influence another.\n"
        f"SHARED (applies to every item):\n{json.dumps(shared, ensure_ascii=False)}\n\n"
        f"ITEMS (index = position, starting at 0):\n{json.dumps(items, ensure_ascii=False)}\n\n"
        f'Answer with {{"items": [...]}} holding exactly {k} results, one per item, in the same order.\n'
    )
    return SYSTEM[spec.engine], user


# ------------------------------------------------------------------------------------------ citizens in words
_DEGREE = {0: "neutral on", 1: "slightly", 2: "clearly", 3: "strongly"}
_POLES = {
    "religion": ("secular", "religious"),
    "economy": ("pro-market", "pro-state"),
    "culture": ("progressive", "conservative"),
    "novelty seeking": ("routine-loving", "novelty-seeking"),
    "harm avoidance": ("bold", "cautious"),
    "reward dependence": ("detached", "approval-seeking"),
    "persistence": ("quick to give up", "persistent"),
    "self-directedness": ("dependent", "self-directed"),
    "cooperativeness": ("competitive", "cooperative"),
    "self-transcendence": ("down-to-earth", "idealistic"),
}
_PILLARS = ("political pluralism", "rule of law", "free press", "property rights", "civil liberties")


def _pole(name: str, v: int) -> str:
    lo, hi = _POLES[name]
    return "neutral between " + lo + " and " + hi if v == 0 else f"{_DEGREE[abs(v)]} {hi if v > 0 else lo}"


def describe_citizen(row: dict) -> dict:
    """The label context of a citizen (integers -3..3) as words, so that a decision model can read it: what each
    number means is part of the prompt, not something the model has to guess."""
    t, c, i, o = row["temperament"], row["character"], row["ideology"], row["opinion"]
    mayor = o[0]
    d = {
        "who": f"{row['age_band']} citizen of the {row['class']} class",
        "ideology": {k: _pole(k, v) for k, v in zip(("religion", "economy", "culture"), i)},
        "temperament": {
            k: _pole(k, v)
            for k, v in zip(("novelty seeking", "harm avoidance", "reward dependence", "persistence"), t)
        },
        "character": {
            k: _pole(k, v) for k, v in zip(("self-directedness", "cooperativeness", "self-transcendence"), c)
        },
        "opinion of the mayor": "neutral"
        if mayor == 0
        else f"{_DEGREE[abs(mayor)]} {'approves' if mayor > 0 else 'disapproves'}",
        "views on weakening the pillars of democracy": {
            n: ("indifferent" if v == 0 else f"{_DEGREE[abs(v)]} {'accepts' if v > 0 else 'rejects'}")
            for n, v in zip(_PILLARS, o[1:])
        },
        "fear of the authorities": "none" if row["fear"] == 0 else ("low" if row["fear"] <= 2 else "high"),
    }
    return d
