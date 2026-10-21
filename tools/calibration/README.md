# Calibration scripts (mocks)

The scripts of the full recalibration of docs/STATUS.md (Session 5). Run from the repository root with `uv run python tools/calibration/<script>`.

- `sweep.py`: one game with parameter overrides and the population/outcome measures used by everything else (`one(...)`).
- `calib.py`: the joint objective (population on the passive baseline, impulse test, outcome bands) and a first LHS over 14 parameters:
  `calib.py 96 1 7` (points, first seed, last seed + 1). Writes `out/calib_top.json`.
- `calib3.py`: LHS with `theta_outcome` tied to the baseline pressure of the council (`to_overrides`), 12 training seeds, validation on 101-124.
- `calib4.py`: refinement of the pillar gate and the capital cost around the best point of `calib3.py`.
- `calib5.py`: wide search used on the mock fitted on a real S1 (see docs/mocks.md); it found no acceptable point.
- `final.py`: before/after table on the validation and hold-out seeds (`out/final.json`).

They are research scripts, not part of the `tle` command: the objective and its weights are in `calib.py::loss`.
