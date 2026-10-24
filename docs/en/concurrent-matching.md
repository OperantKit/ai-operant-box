# Two levers: concurrent VI VI and the matching law

:jp: [日本語版](../ja/concurrent-matching.md)

`ai_operant_box.chamber.concurrent.ConcurrentBox` runs the same SD-R-SR step as `OperantBox` with two operanda (needs `contingency-py`, installed as in the [README](../../README.md#installation)). The reinforcement decision, including the changeover delay (COD; Catania, 1966), is made by `contingency-py`'s `Concurrent` schedule; `concurrent_vi_vi(mean_left, mean_right, cod=2.0, seed=...)` builds `Conc VI VI` with seeded components.

- **Subject interface.** Any `Subject` works. `act()` returns an action label, and `operanda` maps the labels that count as lever presses to operandum keys: by default `"Left Lever"` to `"left"` and `"Right Lever"` to `"right"` (`CONCURRENT_ACTIONS = ("Left Lever", "Right Lever", "Nothing")` is the matching vocabulary for an LLM subject, e.g. `StreamingLLMSubject(client, actions=CONCURRENT_ACTIONS)`). Any other label is not a response; the schedule still gets a time tick. `learn()` receives `"Food Pellet"` or `None` for the lever just pressed.
- **Records.** Each response is a `ChoiceRecord(timestamp, operandum, is_reinforced, cod_suppressed)`.
- **Matching fit.** `ai_operant_box.analytics.matching.choice_totals(records, start=...)` counts responses and reinforcers per lever after a warm-up, and `fit_generalized_matching(b1, b2, r1, r2)` fits `log(B1/B2) = a * log(R1/R2) + log b` (base-10 logs; Baum, 1974) by least squares over conditions, returning `a`, `log_b`, and R². `correction=0.5` adds 0.5 to every count so a short session with an empty cell can still be fitted (report it with the estimates). `changeovers(records, start=...)` counts switches between levers.
- **LLM subject.** `CONCURRENT_SYSTEM_PROMPT` is the two-lever counterpart of the single-lever prompt: it names the actions and the feedback and says nothing about how the levers pay off. `examples/claude_cli_matching.py --model haiku` runs a Claude model through the `claude` CLI on the four ratios below (each condition a fresh session, 100 one-second steps, overall 1 reinforcer per 3 s, no COD by default) and writes the counts, changeovers, and both fits (whole session and second half) to JSON; `--plot <dir>` draws the fits.
- **Reference subject.** `MatchingAgent(sensitivity=s, mean_visit=v)` presses a lever every step, picks the left lever with probability `R_L^s / (R_L^s + R_R^s)` from the reinforcers it has obtained, and redraws its lever with probability `1/v` per step. It exists to check the chamber and the fit, not to model an organism.

```bash
python examples/concurrent_matching.py --cod 0 --sensitivity 1
```

Over four reinforcer ratios (1:1, 1:3, 3:1, 1:9; overall 1 reinforcer per 10 s; 4,000 one-second steps, the first 1,000 dropped) the reference subject gave `a = 1.006`, `log b = 0.015`, R² = 0.998 without a COD, and `a = 0.450` with `sensitivity=0.5`. With a 2 s COD the result depends on the installed `contingency-py`: 0.1.1 discards a reinforcer set up during the COD, the lean lever loses those reinforcers, and the reference subject drifts to exclusive preference (a lever with zero reinforcers makes the fit undefined). A `contingency-py` that holds the reinforcer until the first response after the COD (Herrnstein, 1961) gave `a = 0.966`, `log b = 0.015`, R² = 0.997. The test with a COD is skipped when the installed `contingency-py` discards the reinforcer.

## References

- Baum, W. M. (1974). On two types of deviation from the matching law: Bias and undermatching. *Journal of the Experimental Analysis of Behavior, 22*(1), 231–242. https://doi.org/10.1901/jeab.1974.22-231
- Catania, A. C. (1966). Concurrent operants. In W. K. Honig (Ed.), *Operant behavior: Areas of research and application* (pp. 213–270). Appleton-Century-Crofts.
- Herrnstein, R. J. (1961). Relative and absolute strength of response as a function of frequency of reinforcement. *Journal of the Experimental Analysis of Behavior, 4*(3), 267–272. https://doi.org/10.1901/jeab.1961.4-267
