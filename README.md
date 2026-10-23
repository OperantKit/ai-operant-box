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

A real LLM subject needs no extra when it runs through the `claude` command (see "A real LLM subject without an API key"). To use the Anthropic API instead:

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

To put a language model in the same chamber without an API key, see the next section; with an API key, install the `llm` extra and use `ai_operant_box.agent.llm_agent.LLMSubject` with `AnthropicClient`. To keep the events as an OKL v1 log, install the `canonical` extra and pass them through `ai_operant_box.async_simulator.canonical_sink.mirror_events_to_sink` to a [session-recorder](https://github.com/OperantKit/session-recorder) sink.

## A real LLM subject without an API key

`ClaudeCLIClient` runs the `claude` command (Claude Code) as one resident process and sends each step as one user turn, so the process's conversation is the subject's in-context history. It runs on the logged-in Claude subscription; no API key and no extra are needed, only the `claude` command on `PATH` (log in once by running `claude`). `StreamingLLMSubject` sends only the newest step (`Result: Reward` / `Stimulus: Light ON`), reads the action from the reply, and emits a reply that names no single action as `Nothing` (counted in `invalid_count`; the raw reply stays in `history`).

```python
from ai_operant_box.agent.api_client import ClaudeCLIClient
from ai_operant_box.agent.llm_agent import SYSTEM_PROMPT, StreamingLLMSubject
from ai_operant_box.chamber.environment import OperantBox
from ai_operant_box.chamber.schedule import ContinuousReinforcement

with ClaudeCLIClient(model="haiku", system_prompt=SYSTEM_PROMPT) as client:
    subject = StreamingLLMSubject(client)
    box = OperantBox(subject=subject, schedule=ContinuousReinforcement())
    for _ in range(20):
        box.step(1.0)
print(len(box.records), subject.invalid_count, client.latencies[-1])
```

`examples/claude_cli_crf.py` runs the same session from the command line and writes a cumulative record (PNG, `viz` extra) and a JSON summary with per-step round-trip times.

The system prompt replaces the default Claude Code prompt (`--system-prompt`). With `--append-system-prompt` the coding-assistant prompt stays, and a small model declines the role. Tools, MCP servers, settings files and their hooks, and session persistence are switched off, and the process runs in an empty temporary directory, so no project `CLAUDE.md` reaches the subject. The CLI still adds a one-line identity sentence before the system prompt and short notes with the working directory, the model name, and the date; these are the same in every session.

Measured round-trip time per step on CRF (one machine, October 2026):

| Model alias | Steps | Mean s / step | First step s |
|---|---|---|---|
| `haiku` | 20 | 1.1 to 1.4 | about 4 |
| `haiku` | 150 | 0.8 to 1.1 | about 3 to 4 |
| `sonnet` | 20 | 1.9 | about 2.5 |
| `opus` | 20 | 2.6 | about 3.9 |

The per-step time did not grow over 150 steps with `haiku`. Budget about 1.4 s per step for `haiku`, 2.5 s for `sonnet`, and 3.5 s for `opus`: a 600-step `haiku` session takes about 14 minutes. The `haiku` alias resolved to `claude-haiku-4-5-20251001` at the time of measurement; the CLI decides what each alias means.

**Usage limits.** The calls count against the subscription's usage limits, not an API bill. `client.last_result["total_cost_usd"]` is the conversation's API-equivalent price, a proxy for that usage: 150 `haiku` steps came to about 0.24 USD-equivalent, and the reported seven-day utilisation (`client.rate_limit_info`) did not move at its two-decimal resolution across several hundred `haiku` steps. `sonnet` and `opus` use the limits faster per step. `ClaudeCLIClient.generate()` keeps the stateless `generate(model, system, messages)` interface of `LLMSubject`, but it starts a new process and re-sends the whole history every step, so it is slower and is not the default.

## Reading response probabilities from a local model

Hosted chat APIs return text only, so the probability that a model presses the lever can only be estimated by sampling its reply many times. A local GGUF model run through llama.cpp exposes its next-token logits, so `ai_operant_box.agent.local_logprob` computes the probability of each action directly at every step, without sampling (`local` extra: `llama-cpp-python`).

- `LocalLogprobClient` keeps a ChatML conversation (the chat format of Qwen and many other instruction-tuned GGUF models). `score(text)` adds a user turn and returns, for each action, the log-probability of its whole token sequence as the assistant's reply (product of the token probabilities), the log-probability of its first token alone, and the probabilities renormalised over the action set. `temperature` divides every token's logits before the softmax.
- `LlamaCppBackend` evaluates only the part of each prompt that is new since the previous step, so a 600-step session stays fast.
- `LocalLogprobSubject` runs in `OperantBox` like any other subject. The chamber needs an emitted action to apply the schedule; it is a seeded draw from the computed distribution (`policy="sample"`) or the most probable action (`policy="greedy"`). Each `history` entry keeps the probabilities `p` read at that step.

```bash
.venv/bin/python -m pip install -e ".[local,viz]"
.venv/bin/python examples/local_logprob_fi.py --model path/to/model.gguf --interval 30 --steps 600 --out-dir runs/fi30
```

The script plots P(Lever Press) over the session and its mean by time since the previous reinforcer, and writes every step's probabilities to JSON. With Qwen2.5-1.5B-Instruct (Q4_K_M) on one Apple-silicon machine, 600 FI 30 steps took about 7 minutes (0.7 s per step on average; the prompt grew to about 13,000 tokens). The first tokens of `Lever Press` and `Grooming` are single letters (`L`, `G`) for that tokenizer, which is why the whole-sequence probability, not the first-token probability, is renormalised.

The test that checks the reused prefix against a fresh evaluation runs only when the environment variable `AOB_GGUF` names a GGUF file; the other tests use a fake backend and need no model.

## Two levers: concurrent VI VI and the matching law

`ai_operant_box.chamber.concurrent.ConcurrentBox` runs the same SD-R-SR step as `OperantBox` with two operanda (`concurrent` extra: `contingency-py`). The reinforcement decision, including the changeover delay (COD; Catania, 1966), is made by `contingency-py`'s `Concurrent` schedule; `concurrent_vi_vi(mean_left, mean_right, cod=2.0, seed=...)` builds `Conc VI VI` with seeded components.

- **Subject interface.** Any `Subject` works. `act()` returns an action label, and `operanda` maps the labels that count as lever presses to operandum keys: by default `"Left Lever"` to `"left"` and `"Right Lever"` to `"right"` (`CONCURRENT_ACTIONS = ("Left Lever", "Right Lever", "Nothing")` is the matching vocabulary for an LLM subject, e.g. `StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)`). Any other label is not a response; the schedule still gets a time tick. `learn()` receives `"Food Pellet"` or `None` for the lever just pressed.
- **Records.** Each response is a `ChoiceRecord(timestamp, operandum, is_reinforced, cod_suppressed)`.
- **Matching fit.** `ai_operant_box.analytics.matching.choice_totals(records, start=...)` counts responses and reinforcers per lever after a warm-up, and `fit_generalized_matching(b1, b2, r1, r2)` fits `log(B1/B2) = a * log(R1/R2) + log b` (base-10 logs; Baum, 1974) by least squares over conditions, returning `a`, `log_b`, and R². `correction=0.5` adds 0.5 to every count so a short session with an empty cell can still be fitted (report it with the estimates). `changeovers(records, start=...)` counts switches between levers.
- **LLM subject.** `CONCURRENT_SYSTEM_PROMPT` is the two-lever counterpart of the single-lever prompt: it names the actions and the feedback and says nothing about how the levers pay off. `examples/claude_cli_matching.py --model haiku` runs a Claude model through the `claude` CLI on the four ratios below (each condition a fresh session, 100 one-second steps, overall 1 reinforcer per 3 s, no COD by default) and writes the counts, changeovers, and both fits (whole session and second half) to JSON; `--plot <dir>` draws the fits.
- **Reference subject.** `MatchingAgent(sensitivity=s, mean_visit=v)` presses a lever every step, picks the left lever with probability `R_L^s / (R_L^s + R_R^s)` from the reinforcers it has obtained, and redraws its lever with probability `1/v` per step. It exists to check the chamber and the fit, not to model an organism.

```bash
.venv/bin/python -m pip install -e ".[concurrent]"
.venv/bin/python examples/concurrent_matching.py --cod 0 --sensitivity 1
```

Over four reinforcer ratios (1:1, 1:3, 3:1, 1:9; overall 1 reinforcer per 10 s; 4,000 one-second steps, the first 1,000 dropped) the reference subject gave `a = 1.006`, `log b = 0.015`, R² = 0.998 without a COD, and `a = 0.450` with `sensitivity=0.5`. With a 2 s COD the result depends on the installed `contingency-py`: 0.1.1 discards a reinforcer set up during the COD, the lean lever loses those reinforcers, and the reference subject drifts to exclusive preference (a lever with zero reinforcers makes the fit undefined). A `contingency-py` that holds the reinforcer until the first response after the COD (Herrnstein, 1961) gave `a = 0.966`, `log b = 0.015`, R² = 0.997. The test with a COD is skipped when the installed `contingency-py` discards the reinforcer.

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
| `local` | `llama-cpp-python` | `ai_operant_box.agent.local_logprob` |
| `concurrent` | `contingency-py` | `ai_operant_box.chamber.concurrent` |

Other notes:

- A small `chamber.environment.VirtualClock` (`ManualClock`-style monotonic
  counter) is provided locally, so the chamber runs without
  `experiment-io`'s HAL.
- Single-lever schedule primitives (`chamber.schedule` and `schedules/`) are
  implemented here. `contingency-py` is imported only by
  `chamber.concurrent` (`concurrent` extra).

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
- Baum, W. M. (1974). On two types of deviation from the matching law:
  Bias and undermatching. *Journal of the Experimental Analysis of
  Behavior*, *22*(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Benvenuti, M. F., et al. (2024). The role of schedules of reinforcement in
  the experimental analysis of social behavior. *Perspectives on Behavior
  Science*. https://doi.org/10.1007/s40614-024-00421-5
- Catania, A. C. (1966). Concurrent performances: Reinforcement
  interaction and response independence. *Journal of the Experimental
  Analysis of Behavior*, *9*(3), 253-263. https://doi.org/10.1901/jeab.1966.9-253
- Fleshler, M., & Hoffman, H. S. (1962). A progression for generating
  variable-interval schedules. *Journal of the Experimental Analysis of
  Behavior*, *5*(4), 529-530. https://doi.org/10.1901/jeab.1962.5-529
- Herrnstein, R. J. (1961). Relative and absolute strength of response as
  a function of frequency of reinforcement. *Journal of the Experimental
  Analysis of Behavior*, *4*(3), 267-272. https://doi.org/10.1901/jeab.1961.4-267
- Skinner, B. F. (1938). *The behavior of organisms: An experimental
  analysis*. Appleton-Century.
