# ai-operant-box

:jp: [日本語版 README](README.ja.md)

LLM-as-subject simulator. A large language model acts as a virtual organism
inside an operant chamber, sensing a discriminative stimulus (SD), emitting a
response (R), and receiving a consequence (SR) on every trial — the three-term
contingency of operant conditioning.

## Overview

The package provides:

- **Subjects (`ai_operant_box.agent`)** — abstract `Subject` base, deterministic
  numerical agents (`InductionAgent`, `InductionSubject`, `ScallopingAgent`),
  and an LLM-backed `LLMSubject` whose action policy is in-context history fed
  through Claude.
- **Chambers (`ai_operant_box.chamber`)** — single-subject `OperantBox` and
  multi-subject `LeaderFollowerBox`, each driving a discrete-time SD-R-SR loop.
- **Schedules (`ai_operant_box.chamber.schedule`,
  `ai_operant_box.schedules`)** — CRF / FI plus Fleshler-Hoffman variable and
  random ratio / interval schedules, plus an interdependent dyadic schedule.
- **Analytics (`ai_operant_box.analytics`)** — cumulative recorders, multi-
  session comparison, scallop-model fitter (matplotlib + SciPy import lazily
  inside the analytics functions; not required at import time).
- **CLI (`ai_operant_box.cli`)** — minimal entry point used by smoke tests.

## Install

```bash
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

To enable the LLM-backed subject:

```bash
.venv/bin/python -m pip install -e ".[dev,llm]"
export ANTHROPIC_API_KEY="..."
```

## Run tests

```bash
.venv/bin/python -m pytest -q
```

Tests that exercise the Anthropic SDK are skipped automatically when the
`anthropic` package is not installed.

## Three-term contingency

Code structure mirrors the basic operant unit:

1. **SD (Discriminative Stimulus)** — presented by the `Chamber` to the
   `Subject` at the start of each step.
2. **R (Response)** — emitted by the `Subject` (numerical policy or LLM call).
3. **SR (Consequence)** — delivered by the `Schedule` to the `Subject` whose
   `learn(reinforcer)` updates internal state.

## Local dependency notes

The v0 implementation depended on the sibling packages `contingency-py`,
`experiment-core`, `experiment-io`, `session-recorder`, and `session-analyzer`.
Those dependencies are deferred in this scaffold:

- A small `chamber.environment.VirtualClock` (`ManualClock`-style monotonic
  counter) is provided locally so the chamber can run without
  `experiment-io`'s HAL.
- JSONL event logging is internal and can later be replaced with the
  `session-recorder` `SessionLogger`.
- Schedule primitives are independent (`chamber.schedule` and `schedules/`),
  separate from `contingency-py` for now.

When the surrounding apps stabilise, these shims can be replaced with the
proper packages without touching agent / chamber code.

## Anthropic SDK (opt-in)

The `anthropic` package is an **optional** dependency declared under the
`[llm]` extra. It is imported lazily inside `ai_operant_box.agent.api_client`,
so importing the package or running the deterministic tests does not require
the SDK. API keys are read from the `ANTHROPIC_API_KEY` environment variable
and never hardcoded.

## References

- Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain
  operant performance. *Journal of the Experimental Analysis of Behavior*,
  *113*(2), 390-418. https://doi.org/10.1002/jeab.583
- Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in
  the experimental analysis of social behavior. *Perspectives on Behavior
  Science*. https://doi.org/10.1007/s40614-024-00421-5
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
  variable-interval schedules. *Journal of the Experimental Analysis of
  Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Skinner, B. F. (1938). *The behavior of organisms: An experimental
  analysis*. Appleton-Century.
