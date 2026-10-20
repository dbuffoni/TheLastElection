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


def render(task: str, inp: BaseModel) -> tuple[str, str]:
    spec = TASKS[task]
    user = f"TASK: {task}\n{spec.description}\n\nINPUT:\n{json.dumps(inp.model_dump(), ensure_ascii=False)}\n"
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
