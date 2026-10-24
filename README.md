# ai-operant-box

:jp: [日本語版 README](README.ja.md)

ai-operant-box puts a large language model (LLM), or a simple numerical agent, in a simulated operant chamber in place of an animal. On every step the subject is shown a discriminative stimulus (SD), emits a response (R), and receives the consequence (SR) that the reinforcement schedule arranges: the three-term contingency (Skinner, 1938). It is for researchers and teachers who want to ask whether a model's behavior comes under the control of a schedule, with the procedures and measures used for animals. Within [OperantKit](https://github.com/OperantKit), it is the simulator with LLM subjects; it can write its events as an OKL log (OperantKit's plain-text event log) for the other packages.

- **Same chamber, any subject.** A numerical agent and an LLM run under the same schedule, so a model can be compared with a reference subject.
- **Session time, not API time.** The clock is simulated, so the time a model takes to reply does not count toward an interval.
- **Measures from the animal literature:** the FI scallop fit, FI pattern measures, and the generalized matching law.

## Installation

Python 3.11 or newer. ai-operant-box is not on PyPI yet; install it from GitHub. The core needs only NumPy and runs the numerical agents, the single-lever chamber, and the example below:

```bash
pip install git+https://github.com/OperantKit/ai-operant-box
```

Cumulative records, the scallop fit, OKL logs, and the two-lever chamber use other OperantKit packages, which are not on PyPI either. Install them from GitHub without dependency resolution, then add the third-party libraries:

```bash
G=git+https://github.com/OperantKit
pip install --no-deps $G/contingency-dsl-py $G/contingency-py $G/experiment-core \
  $G/session-recorder $G/session-analyzer
pip install matplotlib scipy
```

pip reports version conflicts between these development builds; they do not affect this package. A Claude model as the subject needs either the `claude` command (Claude Code, logged in) or `pip install anthropic` and `ANTHROPIC_API_KEY`.

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

## What is included

| Part | Contents |
|---|---|
| Subjects (`ai_operant_box.agent`) | The `Subject` interface (`sense(SD)`, `act() -> R`, `learn(SR)`); numerical agents (`ScallopingAgent`, `BernoulliAgent`, `InductionAgent` after Baum & Grace, 2020, `MatchingAgent`); LLM subjects through the Anthropic API, the `claude` command, or a local GGUF model |
| Chambers (`ai_operant_box.chamber`) | `OperantBox` (one lever), `ConcurrentBox` (two levers, concurrent VI VI with a changeover delay, using contingency-py), `LeaderFollowerBox` (two subjects on an interdependent schedule; Benvenuti et al., 2024) |
| Schedules | CRF, EXT, FI, DRL, VI, VR, RI, RR, and a dyadic interdependent schedule |
| Analytics (`ai_operant_box.analytics`) | Cumulative records, the FI scallop fit `P(R \| t) = min(1, k * t**2)`, FI pattern measures (quarter life, index of curvature), and the generalized matching law fit |
| OKL output | `async_simulator.canonical_sink.mirror_events_to_sink` passes the events to a [session-recorder](https://github.com/OperantKit/session-recorder) sink |
| Examples (`examples/`) | Claude subjects on CRF, CRF then extinction, FI, DRL, concurrent VI VI, yoked reinforcement, and a two-subject schedule; numerical subjects on concurrent VI VI; a local model on FI |

How to run an LLM as the subject, with measured time per step: [docs/en/llm-subjects.md](docs/en/llm-subjects.md). The two-lever chamber and the matching-law fit: [docs/en/concurrent-matching.md](docs/en/concurrent-matching.md).

## Limitations

- **Time moves in discrete steps,** usually 1 s; responses within a step are not resolved.
- **Variable schedules are not Fleshler–Hoffman.** VI, VR, RI, and RR values in the single-lever chamber are independent draws from an exponential distribution, rounded to whole numbers. Only the two-lever chamber uses contingency-py's schedules.
- **The subjects are models, not organisms.** The numerical agents exist to check the chamber and the measures. `LLMSubject` takes the model's whole reply as the action, so only a reply that is exactly an action name counts; `StreamingLLMSubject` looks for an action name in the reply and records a reply that names no single action as "Nothing".
- **Hosted LLM subjects are Claude models** (the Anthropic API or the `claude` command). Other models need your own client.
- **The `ai-operant-box` command only runs a short smoke test** (`ai-operant-box smoke`); experiments are run from Python or the scripts in `examples/`.

## For AI coding agents

Guidance for AI coding agents is in [AGENTS.md](AGENTS.md).

## Development

```bash
python -m venv .venv
G=git+https://github.com/OperantKit
.venv/bin/python -m pip install --no-deps -e . $G/contingency-dsl-py $G/contingency-py \
  $G/experiment-core $G/session-recorder $G/session-analyzer
.venv/bin/python -m pip install numpy matplotlib scipy pytest pytest-cov pytest-asyncio
.venv/bin/python -m pytest
```

The anthropic SDK and the `claude` command are not needed for the tests; the one test that loads a local model is skipped unless `llama-cpp-python` is installed and `AOB_GGUF` names a GGUF file. `tests/test_readme_examples.py` runs the 30-second example in both READMEs and checks the printed lines.

## References

- Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain operant performance. *Journal of the Experimental Analysis of Behavior, 113*(2), 390–418. https://doi.org/10.1002/jeab.583
- Benvenuti, M. F., Bernardy, J. L., de Toledo, T. F. N., Santiago, J., Siqueira, J. de O., & Silveira, P. S. P. (2024). The role of schedules of reinforcement in the experimental analysis of social behavior. *Perspectives on Behavior Science, 47*(4), 717–738. https://doi.org/10.1007/s40614-024-00421-5
- Skinner, B. F. (1938). *The behavior of organisms: An experimental analysis*. Appleton-Century.

## License

MIT. See [LICENSE](LICENSE).
