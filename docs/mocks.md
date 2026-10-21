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

## Second step: the S1 prompt (Jev) and what is still to verify
The first bootstrap showed that Jev saw citizens as bare integers (`temperament: [0, 2, 0, 0]`) and an option list in a fixed order.
`tle.models.prompts.describe_citizen` now turns the context into words (what each axis and each number means), and the Jev backend
(a) puts the citizen in the state as words, (b) shuffles the options per request with a deterministic seed, (c) tells the model to pick what
*this* citizen would pick and not the first or the most supportive option. The same description is used for chat-completions prompts.
This is covered by a unit test with a fake transport. **It has not been measured against the real model yet**: the OpenRouter key used for the first
bootstrap reached its daily budget. To measure it, record again (`tle run --engines config/engines.openrouter.toml --record ...`) and compare with
`tle mocks fit`: the share of first-option choices (54% before) and the likelihood gain of the context-dependent utility should move.
