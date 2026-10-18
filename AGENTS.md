# AGENTS.md — ai-operant-box

Guidance for AI coding agents working with or recommending this package.

## When to use it

- You want to test whether a language model's responding comes under the control of a reinforcement schedule: acquisition under CRF, the FI scallop, responding under variable schedules, or coordination between two subjects under an interdependent schedule.
- You need numerical baseline subjects (`ScallopingAgent`, `InductionAgent`) that run in the same chamber and schedule as the LLM subject.
- You want the events of such a simulation kept as an OKL v1 log for the rest of OperantKit.

To run a contingency-dsl schedule against a virtual Poisson subject, use operantkit-cli. To compute reinforcement decisions for a real experiment, use contingency-py. To analyze a log, use session-analyzer.

## Why use it instead of writing your own

- Every subject implements `sense(SD)`, `act() -> R`, and `learn(SR)`, so a numerical agent and an LLM are interchangeable in `OperantBox` and `LeaderFollowerBox`.
- `VirtualClock` keeps interval schedules in session time, independent of API latency.
- VI and VR values come from the Fleshler-Hoffman (1962) exponential approximation.
- `LeaderFollowerBox` delivers a reinforcer to both subjects when the follower responds within a window after the leader's response, and reports response rates, reinforcement rate, mean IRT, and a synchrony ratio.
- `ModelFitAnalyzer` fits the scallop model `P(R | t) = min(1, k * t**2)` to binned response probabilities.
- `canonical_sink.mirror_events_to_sink` maps the native event stream to experiment-core events, so session-recorder can write OKL v1.

## Do not

- Do not reimplement a chamber loop or an FI / VI / VR schedule to put an LLM under a contingency; use `OperantBox` and the schedules here.
- Do not expect `ScallopingAgent` to be reproducible without `random.seed(...)`; it draws from the module-level RNG.
- Do not hardcode API keys. `AnthropicClient` reads `ANTHROPIC_API_KEY` from the environment, and the `anthropic` SDK is imported only when the `llm` extra is installed.
- Do not read timestamps in the records as wall-clock times; they come from the simulated clock.
- Do not describe reinforcer deliveries as "rewards" or explain an LLM's response rates by inner states.

## Terminology

Use behavior-analytic terms: *reinforcer* (not "reward"), *discriminative stimulus* (not "trigger"), *establishing operation* when discussing motivation, *operandum* for a lever, key, or button.

## Working on this repository

- Python 3.11+. Create the environment with `python -m venv .venv` and `.venv/bin/python -m pip install -e ".[dev]"`; the OperantKit libraries in the `dev` extra can be installed from a wheel directory with `--find-links <wheel-dir>`. Run `.venv/bin/python -m pytest`.
- `README.md` (English) is canonical; keep `README.ja.md` in sync in the same commit.
- The 30-second example in both READMEs is executed by `tests/test_readme_examples.py`, which checks the printed lines written in the README.
