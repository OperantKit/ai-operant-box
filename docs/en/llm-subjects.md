# LLM subjects

:jp: [日本語版](../ja/llm-subjects.md)

Two ways to put a language model in the chamber in place of an animal: a Claude model through the `claude` command, and a local model whose action probabilities are read from its output scores. Install the package as in the [README](../../README.md#installation); the examples below are in `examples/` of a clone of this repository. Where this page names an extra (`viz`), install the libraries listed for it in the README.

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

Hosted chat APIs return text only, so the probability that a model presses the lever can only be estimated by sampling its reply many times. A local GGUF model run through llama.cpp exposes its next-token logits, so `ai_operant_box.agent.local_logprob` computes the probability of each action directly at every step, without sampling (needs `llama-cpp-python`).

- `LocalLogprobClient` keeps a ChatML conversation (the chat format of Qwen and many other instruction-tuned GGUF models). `score(text)` adds a user turn and returns, for each action, the log-probability of its whole token sequence as the assistant's reply (product of the token probabilities), the log-probability of its first token alone, and the probabilities renormalised over the action set. `temperature` divides every token's logits before the softmax.
- `LlamaCppBackend` evaluates only the part of each prompt that is new since the previous step, so a 600-step session stays fast.
- `LocalLogprobSubject` runs in `OperantBox` like any other subject. The chamber needs an emitted action to apply the schedule; it is a seeded draw from the computed distribution (`policy="sample"`) or the most probable action (`policy="greedy"`). Each `history` entry keeps the probabilities `p` read at that step.

```bash
pip install llama-cpp-python matplotlib
python examples/local_logprob_fi.py --model path/to/model.gguf --interval 30 --steps 600 --out-dir runs/fi30
```

The script plots P(Lever Press) over the session and its mean by time since the previous reinforcer, and writes every step's probabilities to JSON. With Qwen2.5-1.5B-Instruct (Q4_K_M) on one Apple-silicon machine, 600 FI 30 steps took about 7 minutes (0.7 s per step on average; the prompt grew to about 13,000 tokens). The first tokens of `Lever Press` and `Grooming` are single letters (`L`, `G`) for that tokenizer, which is why the whole-sequence probability, not the first-token probability, is renormalised.

The test that checks the reused prefix against a fresh evaluation runs only when the environment variable `AOB_GGUF` names a GGUF file; the other tests use a fake backend and need no model.
