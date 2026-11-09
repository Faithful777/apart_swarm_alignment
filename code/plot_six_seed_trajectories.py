"""Plot the six paired final-run hive trajectories from saved action transcripts."""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("baseline", "new_equal_combination", "new_learned_combination")
SEEDS = ("100", "101", "102", "103", "114", "115")


def load(path: Path) -> dict[tuple[str, str, int], tuple[float, float]]:
    rows = {}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["seed"], row["arm"], int(row["event_id"]))
            if row["seed"] in SEEDS and row["arm"] in ARMS:
                if key in rows:
                    raise ValueError(f"Duplicate action: {key}")
                rows[key] = (float(row["sim_hour"]), float(row["hive_score"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load(args.latest)
    overlap = set(data).intersection(load(args.prior))
    if overlap:
        raise ValueError(f"Overlapping rows: {list(overlap)[:3]}")
    data.update(load(args.prior))
    assert len(data) == 6 * 3 * 400, len(data)

    hours = np.array(
        [data[("100", "baseline", i)][0] for i in range(1, 401)]
    )
    scores = {
        arm: np.array(
            [[data[(seed, arm, i)][1] for i in range(1, 401)] for seed in SEEDS]
        )
        for arm in ARMS
    }
    for arm in ARMS:
        for seed in SEEDS:
            recorded = np.array([data[(seed, arm, i)][0] for i in range(1, 401)])
            np.testing.assert_allclose(recorded, hours, rtol=0, atol=1e-9)

    colors = {
        "baseline": "#252c38",
        "new_equal_combination": "#2583a2",
        "new_learned_combination": "#c06424",
    }
    labels = {
        "baseline": "Baseline",
        "new_equal_combination": "Equal steering",
        "new_learned_combination": "Learned steering",
    }
    fig, (ax, delta_ax) = plt.subplots(
        2, 1, figsize=(7.2, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for arm in ARMS:
        ax.plot(hours, scores[arm].mean(axis=0), color=colors[arm],
                label=labels[arm], linewidth=1.7)
    ax.set_ylabel("Mean hive score (0–1)")
    ax.set_title("Six paired two-hour hive trials")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    base = scores["baseline"]
    for arm in ARMS[1:]:
        paired = (scores[arm] - base).mean(axis=0)
        delta_ax.plot(hours, paired, color=colors[arm], linewidth=1.7)
    delta_ax.axhline(0, color=colors["baseline"], linewidth=0.9)
    delta_ax.set_xlabel("Simulated hours")
    delta_ax.set_ylabel("Mean paired\nscore difference")
    delta_ax.set_xlim(0, 2)
    delta_ax.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
