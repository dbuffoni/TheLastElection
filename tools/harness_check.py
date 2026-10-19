#!/usr/bin/env python3
"""Checks the Claude Code harness itself: every skill/agent has frontmatter with name+description, names match
their directory/file, and settings.json is valid JSON whose hook scripts exist."""

import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
errs = []


def front(path: Path) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.S)
    if not m:
        errs.append(f"{path}: missing frontmatter")
        return {}
    return dict(re.findall(r"^(\w[\w-]*):\s*(.*)$", m.group(1), re.M))


for sk in sorted((root / ".claude" / "skills").glob("*/SKILL.md")):
    fm = front(sk)
    if fm.get("name") != sk.parent.name:
        errs.append(f"{sk}: name '{fm.get('name')}' != directory '{sk.parent.name}'")
    if len(fm.get("description", "")) < 20:
        errs.append(f"{sk}: description too short")
for ag in sorted((root / ".claude" / "agents").glob("*.md")):
    fm = front(ag)
    if fm.get("name") != ag.stem:
        errs.append(f"{ag}: name != file name")
    if not fm.get("description"):
        errs.append(f"{ag}: no description")
st = json.loads((root / ".claude" / "settings.json").read_text())
for h in st.get("hooks", {}).get("PostToolUse", []):
    for hk in h["hooks"]:
        script = hk["command"].split()[-1]
        if not (root / script).exists():
            errs.append(f"hook script missing: {script}")
print("\n".join(errs) if errs else "harness ok")
sys.exit(1 if errs else 0)
