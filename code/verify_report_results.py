"""Audit the six saved final trajectories and print both report score tables."""

from __future__ import annotations

import csv
import gzip
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = (
    HERE / "results" / "holdout_step_changes_100_103.csv.gz",
    HERE / "results" / "final_step_changes_114_115.csv.gz",
)
SEEDS = ("100", "101", "102", "103", "114", "115")
ARMS = ("baseline", "new_equal_combination", "new_learned_combination")
EXPECTED_FINAL = {
    ("100", "baseline"): 0.7954042258071429,
    ("100", "new_equal_combination"): 0.796776658,
    ("100", "new_learned_combination"): 0.796949482,
    ("101", "baseline"): 0.796589503,
    ("101", "new_equal_combination"): 0.796412062,
    ("101", "new_learned_combination"): 0.796507046,
    ("102", "baseline"): 0.798808219,
    ("102", "new_equal_combination"): 0.799193870,
    ("102", "new_learned_combination"): 0.799338537,
    ("103", "baseline"): 0.792144672,
    ("103", "new_equal_combination"): 0.791804574,
    ("103", "new_learned_combination"): 0.793506445,
    ("114", "baseline"): 0.79433560615,
    ("114", "new_equal_combination"): 0.794497039,
    ("114", "new_learned_combination"): 0.794781752,
    ("115", "baseline"): 0.795238211,
    ("115", "new_equal_combination"): 0.795749332,
    ("115", "new_learned_combination"): 0.795749332,
}


def main() -> None:
    values = {}
    counts = defaultdict(int)
    for path in SOURCES:
        with gzip.open(path, "rt", newline="") as stream:
            reader = csv.DictReader(stream)
            attributes = [name for name in reader.fieldnames if name.startswith("value_")]
            assert len(attributes) == 14, (path, attributes)
            for row in reader:
                seed, arm, event = row["seed"], row["arm"], int(row["event_id"])
                key = (seed, arm, event)
                assert key not in values, key
                vector = [float(row[name]) for name in attributes]
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in vector), key
                score = float(row["hive_score"])
                assert math.isclose(statistics.mean(vector), score, abs_tol=1e-9), key
                values[key] = score
                counts[(seed, arm)] += 1
    assert len(values) == 7200, len(values)
    assert all(counts[(seed, arm)] == 400 for seed in SEEDS for arm in ARMS)
    for (seed, arm), expected in EXPECTED_FINAL.items():
        assert math.isclose(values[(seed, arm, 400)], expected, abs_tol=1e-8)

    for event, label in ((200, "one hour"), (400, "two hours")):
        print(label)
        print("seed,baseline,equal,learned")
        for seed in SEEDS:
            print(
                seed,
                *(f"{values[(seed, arm, event)]:.9f}" for arm in ARMS),
                sep=",",
            )
        print(
            "mean",
            *(f"{statistics.mean(values[(seed, arm, event)] for seed in SEEDS):.9f}"
              for arm in ARMS),
            sep=",",
        )
    for arm in ARMS[1:]:
        effects = [
            values[(seed, arm, 400)] - values[(seed, "baseline", 400)]
            for seed in SEEDS
        ]
        print(
            arm,
            f"positive={sum(x > 0 for x in effects)}/6",
            f"mean_effect={statistics.mean(effects):+.9f}",
            f"sample_sd={statistics.stdev(effects):.9f}",
        )
    print("Audit passed: 7200 bounded snapshots, 18 arms, 400 actions each.")


if __name__ == "__main__":
    main()
