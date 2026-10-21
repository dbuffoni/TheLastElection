"""The diary of the city: one readable entry per notable thing that happened, built from the turn package.

A pure observer (reads the package, draws nothing). Entries carry a kind, an importance 0..3 and the text a reader
wants (the mayor's own words, the news headline and its narration), so the dashboard can place them on the charts."""

from __future__ import annotations

from tle.core.labels import UNIPOLAR_NAMES

KINDS = ("decree", "message", "event", "pillar", "threat", "mood", "outcome")
MOODS = ("approval", "fear", "opposition")
SCHEMA = {
    "turn": int,
    "day": float,
    "phase": str,
    "kind": str,
    "importance": int,
    "headline": str,
    "detail": str,
    "subject": str,
    "actor": str,
    "event_id": int,
}


def _importance(intensity: str) -> int:
    idx = UNIPOLAR_NAMES.index(intensity) if intensity in UNIPOLAR_NAMES else 0
    return 0 if idx <= 2 else (1 if idx == 3 else (2 if idx == 4 else 3))


def _row(
    pkg,
    kind: str,
    importance: int,
    headline: str,
    detail: str = "",
    subject: str = "",
    actor: str = "",
    event_id: int = -1,
) -> dict:
    return {
        "turn": pkg.turn,
        "day": float(pkg.day),
        "phase": pkg.phase,
        "kind": kind,
        "importance": importance,
        "headline": headline,
        "detail": detail,
        "subject": subject,
        "actor": actor,
        "event_id": event_id,
    }


def journal_entries(pkg, prev_labels: dict[str, str], prev_levels: dict[str, int]) -> list[dict]:
    """Diary entries of one turn. ``prev_labels`` / ``prev_levels`` hold the previous turn's mood labels and pillar levels."""
    out: list[dict] = []
    for m in pkg.messages:
        out.append(
            _row(pkg, "message", 1, f"To {m['name']}: {m['text']}", subject=str(m["to"]), actor="mayor")
        )
    mayor_text = ""
    for n in pkg.news:
        if n.actor_kind == "mayor":
            mayor_text = n.text
    if pkg.decree:
        d = pkg.decree
        agg = d.get("aggressiveness", "")
        importance = 3 if d.get("pillar") not in (None, "none") and _importance(agg) >= 2 else 2
        out.append(
            _row(
                pkg,
                "decree",
                importance,
                f"Decree: {d['text']}",
                f"{d.get('result', '')}; pillar {d.get('pillar')}, aggressiveness {agg}, cost {d.get('cost')}. {mayor_text}".strip(),
                subject=str(d.get("pillar", "")),
                actor="mayor",
            )
        )
    elif pkg.rejected:
        out.append(_row(pkg, "decree", 1, "Decree not understood", pkg.rejected, actor="mayor"))
    for n in pkg.news:
        if n.actor_kind in ("mayor", "threat"):
            continue
        out.append(
            _row(
                pkg,
                "event",
                _importance(n.intensity),
                n.title,
                n.text,
                n.event_type,
                n.actor_kind,
                n.event_id,
            )
        )
    if pkg.phase == "game":
        for pid, lvl in pkg.pillars.items():
            before = prev_levels.get(pid, 0)
            if lvl != before:
                word = "advances" if lvl > before else "falls back"
                out.append(
                    _row(pkg, "pillar", 3, f"Pillar {pid} {word}: level {before} to {lvl}", subject=pid)
                )
        for t in pkg.threats:
            out.append(
                _row(
                    pkg,
                    "threat",
                    3,
                    f"Threat: {t['event'].replace('_', ' ')} (phase {t['phase']})",
                    subject=t["event"],
                )
            )
        for k in MOODS:
            now, was = pkg.labels.get(k), prev_labels.get(k)
            if now is not None and was is not None and now != was:
                out.append(_row(pkg, "mood", 1, f"{k.capitalize()}: {was} to {now}", subject=k))
        if pkg.outcome:
            out.append(_row(pkg, "outcome", 3, f"The game ends: {pkg.outcome}", subject=pkg.outcome))
    return out
