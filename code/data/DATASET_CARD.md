# Bee decision SFT dataset

This generated dataset contains balanced, one-turn synthetic observations paired
with physically executable action proposals. Inputs use the bee simulator's
current local observation and system prompt. Completions are strict six-field
action JSON.

Labels come from transparent hand-written scenario recipes, not GPT distillation,
observed colonies, expert annotation, or optimal-policy search. Each target is
checked by the simulator in the state included in that example. Classes are
intentionally balanced and do not represent real-world action rates. Held-out
splits test interpolation across randomized instances of the same scenario
families, not generalization to biological hives.

The training objective is to imitate these local decision heuristics. It cannot
establish biological correctness. Review with bee experts, compare to validated
baselines, and measure execution-time schema/precondition success before using
the adapter in a research run.
