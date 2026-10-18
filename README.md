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

From a clone, the core package needs only NumPy:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

This is enough for the numerical agents, the chambers, the schedules, and
the `ai-operant-box` command. The development extra adds the test tools
and the optional OperantKit libraries (session-analyzer, experiment-core,
session-recorder). If those libraries are not yet available from your
package index, build or download their wheels into one directory and
point pip at it:

```bash
.venv/bin/python -m pip install --find-links <wheel-dir> -e ".[dev]"
```

To enable the LLM-backed subject:

```bash
.venv/bin/python -m pip install -e ".[dev,llm]"
export ANTHROPIC_API_KEY="..."
```

## 30-second example

Put a numerical subject whose response probability grows with the square of the time since the last reinforcer (`P(R) = min(1, k * t**2)`) on a fixed-interval 30-s schedule for 600 one-second steps. No LLM and no API key are needed:

```python
import random

from ai_operant_box.agent.scallop_agent import ScallopingAgent
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import FixedInterval

random.seed(42)  # ScallopingAgent draws from the module-level RNG
agent = ScallopingAgent(k=0.005)
box = OperantBox(subject=agent, schedule=FixedInterval(30.0))

for _ in range(600):  # 600 one-second steps of FI 30 s
    agent.update_time(1.0)
    box.step(1.0)

responses = len(box.records)
reinforcers = sum(r.is_reinforced for r in box.records)
print(f"responses={responses} reinforcers={reinforcers}")
# responses=422 reinforcers=20

# Responses in each tenth of the 30-s interval, pooled over the session
bins = [0] * 10
last_sr = 0.0
for r in box.records:
    bins[min(9, int((r.timestamp - last_sr) / 3.0))] += 1
    if r.is_reinforced:
        last_sr = r.timestamp
print(bins)
# [1, 5, 12, 30, 54, 60, 60, 60, 60, 80]
```

Responding is rare early in each interval and rises toward the end (the FI scallop); the last tenth also holds the reinforced response of each interval. The clock is simulated, so the run takes well under a second and prints the same numbers every time.

To put a language model in the same chamber, install the `llm` extra and use `ai_operant_box.agent.llm_agent.LLMSubject`. To keep the events as an OKL v1 log, install the `canonical` extra and pass them through `ai_operant_box.async_simulator.canonical_sink.mirror_events_to_sink` to a [session-recorder](https://github.com/OperantKit/session-recorder) sink.

## Why not a 50-line hand-written chamber?

A short loop that asks a model "press or not" and counts presses leaves out what makes the result comparable with animal data:

- **The three-term contingency as an interface.** Every subject implements `sense(SD)`, `act() -> R`, and `learn(SR)` (`ai_operant_box.agent.base.Subject`), so a numerical agent and an LLM run in the same chamber and the same schedule.
- **A simulated clock.** `VirtualClock` advances by the step size, so interval schedules are timed in session time, not in wall-clock time spent waiting for an API.
- **Variable schedules from the Fleshler-Hoffman (1962) progression.** VI and VR values come from the Fleshler-Hoffman exponential approximation; RI and RR are provided separately.
- **A dyadic interdependent schedule.** `LeaderFollowerBox` delivers a reinforcer to both subjects when the follower responds within a window after the leader's response, and reports response rates, reinforcement rate, mean IRT, and a synchrony ratio.
- **The FI scallop model.** `ai_operant_box.analytics.model_fit.ModelFitAnalyzer` fits `P(R | t) = min(1, k * t**2)` to binned response probabilities.
- **In-context learning history for an LLM subject.** `LLMSubject` sends each earlier stimulus, response, and consequence back to the model as conversation turns.

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

## Optional extras and OperantKit libraries

The core package depends only on NumPy. Other OperantKit libraries are
optional and imported lazily:

| Extra | Installs | Used by |
|---|---|---|
| `viz` | `session-analyzer`, matplotlib, SciPy | `ai_operant_box.analytics` (cumulative records, scallop fit) |
| `canonical` | `experiment-core`, `session-recorder` | `ai_operant_box.async_simulator.canonical_sink` (mirror the native event stream to OKL v1) |
| `llm` | `anthropic` | `ai_operant_box.agent.api_client` |

Other notes:

- A small `chamber.environment.VirtualClock` (`ManualClock`-style monotonic
  counter) is provided locally, so the chamber runs without
  `experiment-io`'s HAL.
- Schedule primitives (`chamber.schedule` and `schedules/`) are implemented
  here. This package does not import `contingency-py`.

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
