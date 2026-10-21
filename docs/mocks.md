# Mocks of S1 and S2

The simulated engines (`SimulatedEngine`) stand in for the real models while balancing; the real ones validate. This page
says what the mocks know and how to improve them with data from a real model.

## What the mocks do
- **S1 choices** (`choose_reaction`): utility `u = -d(s_hat, s_k) - lambda_HA * HA * e_k + beta_support * stance_actor + tau * Gumbel`
  over the labels of the citizen and the extracted features of the options (`SimulatedEngine.choice_utilities`).
- **S1 decree classification** (free text, `tle play`): keyword rules for the pillar, the domain, the aggressiveness, the
  predicate and the transfers, plus a **decree bank**: decrees labelled by a real S1 (`config/mocks/decree_bank.toml`). A decree
  close to a labelled one gets its labels; labelled neighbours correct the pillar, the domain and the aggressiveness of the rules.
- **S2 narration**: templates per event type, per intensity and per stance (`config/mocks/narration.toml`); no numbers.

## Bootstrapping from a real model
```bash
# 1. Label the battery of decrees with the real S1 (Jev) and write the bank
uv run tle mocks label-decrees --engines config/engines.openrouter.toml
uv run tle mocks eval                       # agreement of the mock with those labels (rules alone, and with the bank held out)

# 2. Record the choices of a real S1 in a short game (the warm-up runs on simulated engines)
uv run tle run --engines config/engines.openrouter.toml --strategy rentier --n 80 --max-turns 30 \
    --record runs/mocks/records-rentier.jsonl

# 3. Fit the simulated choice on the recorded choices
uv run tle mocks fit runs/mocks/records-*.jsonl --out config/mocks/s1_fit.toml
```
`tle mocks fit` maximizes the likelihood of the real choices under the utility model (`lambda_HA`, `tau_sim`, `w_actor_stance`,
`beta_support`; the first and third never fit to zero, so the mechanisms stay alive). The result is written, not applied:
promote values by editing `config/params/technical.toml` after reading the report.

## What the first bootstrap found (Jev as S1, a small open LLM as S2)
- 135 events, 4911 real choices: the default mock scores below a uniform choice (log-likelihood -12827 against -11308); the fitted
  model scores -10479 and agrees with the real argmax in 48% of the choices.
- The real choices are **position-biased and supportive**: 54% pick the first option and the actor stance of the picked options is
  support in 74% of the cases; the labels of the citizen explain little. This is a property of the real pipeline (prompt and
  option order), not of the mock, so the mock defaults were **not** moved to the fit: balance on a pipeline that ignores the
  citizen would be balance on a bug. Improving the S1 prompt (the citizen's labels first, shuffled options) comes first.

## Second step: the S1 prompt (Jev), measured
The first bootstrap showed that Jev saw citizens as bare integers and an option list in a fixed order. `tle.models.prompts.describe_citizen` now turns the
context into words, the Jev backend puts the citizen in the state as words, shuffles the options per request with a deterministic seed and tells the model to pick
what *this* citizen would pick. Measured on three short games with the real Jev (80 events, 2426 choices, about 0.5 dollars):

| | first bootstrap | new prompt |
|---|---|---|
| picks the first option | 54% | 35% |
| log-likelihood against a uniform choice | +7% | +23% |
| mayor's opponents who pick an option that sides with the mayor | not measurable | 5% (36% of the options do) |
| neutral / approving citizens who do | not measurable | 42% / 98% |

So the real S1 is context-sensitive, and what it does is **echo**: every citizen reinforces the opinion they already have. The fit of the simulated choice
(`config/mocks/s1_fit.toml`: `lambda_HA` 0.05, `tau_sim` 0.43, `w_actor_stance` 0.5, `beta_support` 3.0, `support_gate` 1, `w_pillar_stance` 0.25) describes it; a gated support preference (who already approves likes to say so) fits much better than a flat one.

**Consequence for the balance.** Putting those values in the mock breaks the calibration: the objective of `tools/calibration` goes from 0.47 to 9.8 (all strategies
win, assault is never removed, a passive mayor is removed in 79% of the seeds, the impulse test gives 0.56 instead of 0.33), and a wide search over 16 parameters
(including the frequency of the events of factions and influencers) does not find a point with an assault removal above 0. With echo, a mayor who starts with a
majority keeps it, and opinions swing as a block when an actor event arrives. The calibrated defaults therefore stay on the previous mock parameters; the fit is
a profile to use with `--set lambda_HA=0.05 --set tau_sim=0.43 --set w_actor_stance=0.5 --set beta_support=3 --set support_gate=1 --set w_pillar_stance=0.25`. Making the game winnable *and* losable under echo is
a design question (what counterweights an echo chamber: decay of opinion toward the base, cross-pressure from ties, a stronger direct effect of decrees), not a tuning one.

## Third step: cross-pressure and what it did not fix
The context of a citizen now includes the opinion of the people they know (`gamma_social` weighs it in the simulated choice; the Jev prompt says it in words, so a
new `--record` stores it and `tle mocks fit` can fit `gamma_social`; the recorded data used here predate it). It did not rebalance the fitted profile: the swings
of the mayor's opinion came from common shocks (events that reached every adult at full strength), now fixed (STATUS, Session 7), and from a real S1 that accepts
attacks on the pillars. Under the fitted profile the approval of the mayor still grows with every decree. The open design question stays open: with the pipeline
as it is, the game is winnable by attacking, and a counterweight has to come from the rules, not from the choice of the mock.
