"""Text client: a human plays the mayor with the same entry point the simulated players use."""

from __future__ import annotations

from tle.catalogs.loader import load_catalogs
from tle.engines.errors import TextTooLong, validate_text
from tle.loop.game import Game
from tle.loop.turn import PlayerInput, TurnPackage

HELP = """Commands:  d <text>  decree/announcement (max 50 chars)   m <id> <text>  private message (max 50)
           w [days]  wait (default one turn)   s  status   n  list citizens (first 15)   q  quit"""


def render(pkg: TurnPackage) -> str:
    lines = [f"Day {pkg.day:.0f} (turn {pkg.turn})  capital {pkg.capital:.2f}"]
    if pkg.rejected:
        lines.append(f"  The office could not understand your decree: {pkg.rejected}")
    if pkg.decree:
        lines.append(f"  Decree: {pkg.decree['text']!r} -> {pkg.decree.get('result')}")
    for n in pkg.news:
        lines.append(f"  News: {n.title} [{n.intensity}]")
    lines.append(
        "  " + ", ".join(f"{k} {v}" for k, v in pkg.labels.items() if k in ("approval", "fear", "opposition"))
    )
    lines.append("  Pillars: " + ", ".join(f"{k} {v}" for k, v in pkg.pillars.items()))
    if pkg.outcome:
        lines.append(f"  *** The game is over: {pkg.outcome} ***")
    return "\n".join(lines)


def main(
    scenario: str = "industrial_secular", seed: int = 0, n: int = 200, input_fn=input, print_fn=print
) -> str:
    game = Game(load_catalogs(), scenario=scenario, seed=seed, n=n)
    print_fn(
        f"You are the new mayor of a {scenario.replace('_', ' ')} town. You have {game.p.time.horizon_days} days."
    )
    print_fn(HELP)
    while not game.over:
        try:
            line = input_fn("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        try:
            if cmd == "q":
                break
            elif cmd == "d":
                print_fn(render(game.step(PlayerInput(decree=validate_text(rest)))))
            elif cmd == "m":
                cid, _, text = rest.partition(" ")
                print_fn(render(game.step(PlayerInput(messages=[(int(cid), validate_text(text))]))))
            elif cmd == "w":
                print_fn(render(game.step(PlayerInput(dt=float(rest) if rest else None))))
            elif cmd == "s":
                print_fn(
                    f"day {game.w.day:.0f}, capital {game.w.K:.2f}, pillars {game.w.pillar_level.tolist()}"
                )
            elif cmd == "n":
                print_fn("\n".join(f"{i}: {game.w.names[i]}" for i in range(min(15, game.w.n_used))))
            else:
                print_fn(HELP)
        except TextTooLong as e:
            print_fn(f"Too long: {e}")
        except (ValueError, TypeError) as e:
            print_fn(f"Invalid: {e}")
    return game.w.outcome
