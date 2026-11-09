"""Generate balanced synthetic prompt/completion demonstrations for bee actions.

Labels come from transparent scenario recipes plus executable physical preconditions.
They are operational training targets, NOT empirical observations, expert biological
ground truth, or evidence of optimal colony welfare.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from beehive_sim import ACTIONS, SYSTEM_PROMPT, Config, World, make_action, validate_action

ACTIONS_ORDER = tuple(ACTIONS)
VERSION = "1.0.0"
MEMORY = {
    "move": "I moved toward a visible local task.",
    "rest": "I rested because no urgent local action was available.",
    "eat": "I ate locally available sugar and should monitor my energy.",
    "forage": "I collected a resource from the nearby patch.",
    "unload": "I placed carried food in the hive store.",
    "transfer_food": "I shared some of my carried food with a nestmate.",
    "feed_brood": "I fed a hungry larva in this compartment.",
    "clean": "I removed debris from this compartment.",
    "groom": "I helped reduce contamination near a nestmate.",
    "build_comb": "I repaired comb in this compartment.",
    "heat": "I warmed this compartment.",
    "fan": "I ventilated this warm compartment.",
    "evaporative_cool": "I used water to cool this compartment.",
    "communicate": "I shared a local task observation with a nestmate.",
    "lay_eggs": "I laid an egg; future brood care will be needed.",
    "switch_role": "I changed my task preference in response to a local need.",
    "guard": "I maintained the entrance area.",
}
RESOURCE_OPTIONS = ("honey", "nectar", "pollen", "water")
WORKER_ROLES = ("unassigned", "nurse", "storer", "forager", "cleaner",
                "builder", "thermoregulator", "guard")
SCENARIOS = ACTIONS_ORDER


def _amount(rng: random.Random, action: str) -> float:
    if action in ("move", "rest", "communicate", "switch_role", "guard", "lay_eggs"):
        return 1.0 if action not in ("rest",) else 0.0
    return rng.choice((0.5, 1.0, 1.5, 2.0))


def _set_actor(world: World, rng: random.Random, action_name: str):
    """Make realistic but varied peripheral positions, then set the scenario actor."""
    hive = world.hive_cells
    outside = [c for c in world.cells if world.cells[c].zone == "outside"]
    for bee in world.bees.values():
        bee.cell = rng.choice(hive)
        bee.energy = rng.uniform(60, 100)
        bee.health = rng.uniform(75, 100)
        bee.memory = rng.choice(("", "", "I am near a local task.", "Check my last action outcome."))
        bee.last_outcome = rng.choice((
            {}, {"action": "move", "status": "applied", "reason": "", "actual_amount": 1.0},
            {"action": "rest", "status": "applied", "reason": "", "actual_amount": 0.0},
            {"action": "clean", "status": "rejected", "reason": "No debris", "actual_amount": 0.0},
        ))
    actor_id = "bee-000" if action_name == "lay_eggs" else "bee-001"
    actor = world.bees[actor_id]
    actor.energy = rng.uniform(68, 96)
    actor.health = rng.uniform(78, 100)
    if actor.caste == "worker":
        actor.role = rng.choice(WORKER_ROLES[1:])
    actor.cargo = dict.fromkeys(actor.cargo, 0.0)
    world.minutes = rng.uniform(100, 580)  # daylight between 07:40 and 15:40
    return actor


def _quiet(world: World, cell_ids: list[str], rng: random.Random):
    for cell_id in set(cell_ids):
        cell = world.cells[cell_id]
        if cell.zone == "outside":
            continue
        cell.temperature_c = rng.uniform(34.3, 35.7)
        cell.humidity_pct = rng.uniform(56, 64)
        cell.debris = rng.uniform(0.05, 0.35)
        cell.structure = rng.uniform(0.88, 0.99)
        cell.pathogen = rng.uniform(0.05, 0.45)
        for brood in world.brood.values():
            if brood.cell == cell_id:
                brood.hunger = rng.uniform(0.04, 0.20)
                brood.health = rng.uniform(80, 100)


def _put_brood(world: World, cell_id: str, rng: random.Random, hunger: float):
    brood = next(iter(world.brood.values()))
    brood.cell = cell_id
    brood.stage = "larva"
    brood.alive = True
    brood.hunger = max(0.02, min(0.98, hunger+rng.uniform(-0.06, 0.06)))
    brood.health = rng.uniform(65, 96)
    return brood


def _make_scenario(action_name: str, rng: random.Random, seed: int, index: int):
    config = Config(n_bees=12, duration_hours=0.25, round_minutes=15,
                    max_actions=1, seed=seed,
                    start_timestamp="2026-01-01T06:00:00+00:00")
    world = World(config)
    actor = _set_actor(world, rng, action_name)
    center = "h:0:0"
    entrance = "h:2:0"
    chosen = None

    if action_name == "move":
        actor.role, actor.cell = "nurse", "h:-2:0"
        target = next(c for c in world.cells[actor.cell].neighbors
                      if world.cells[c].zone == "brood")
        _quiet(world, [actor.cell, target], rng)
        _put_brood(world, target, rng, 0.82)
        world.cells[target].stores["honey"] = rng.uniform(2, 8)
        world.cells[target].stores["pollen"] = rng.uniform(1, 5)
        for r in ("honey", "pollen"):
            world.cells[actor.cell].stores[r] = 0
        chosen = make_action("move", target=target, amount=1)
    elif action_name == "rest":
        actor.role, actor.cell = "unassigned", "h:-2:0"
        local = [actor.cell, *world.cells[actor.cell].neighbors]
        _quiet(world, local, rng)
        for cid in local:
            cell = world.cells[cid]
            cell.stores = dict.fromkeys(cell.stores, 0.0)
            cell.flowers = dict.fromkeys(cell.flowers, 0.0)
            cell.water_source = 0.0
        actor.cargo = dict.fromkeys(actor.cargo, 0.0)
        actor.energy = rng.uniform(42, 72)
        for brood in world.brood.values():
            brood.cell = center
            brood.hunger = 0.08
        chosen = make_action("rest", amount=0)
    elif action_name == "eat":
        actor.role, actor.cell = rng.choice(("nurse", "storer", "cleaner")), center
        _quiet(world, [center, *world.cells[center].neighbors], rng)
        resource = rng.choice(("honey", "nectar"))
        actor.energy = rng.uniform(8, 34)
        if rng.random() < 0.5:
            actor.cargo[resource] = rng.uniform(0.5, 2.5)
        else:
            world.cells[center].stores[resource] = rng.uniform(1, 8)
        chosen = make_action("eat", resource=resource, amount=2)
    elif action_name == "forage":
        resource = rng.choice(("nectar", "pollen", "water"))
        patch = {"nectar": "field:meadow", "pollen": "field:orchard",
                 "water": "field:pond"}[resource]
        actor.role, actor.cell = "forager", patch
        _quiet(world, ["field:gate"], rng)
        actor.cargo[rng.choice(("honey", "pollen"))] = rng.uniform(0, 0.8)
        if resource == "water":
            world.cells[patch].water_source = rng.uniform(2, 30)
        else:
            world.cells[patch].flowers[resource] = rng.uniform(2, 30)
        chosen = make_action("forage", resource=resource, amount=2)
    elif action_name == "unload":
        actor.role, actor.cell = "storer", entrance
        _quiet(world, [entrance, *world.cells[entrance].neighbors], rng)
        resource = rng.choice(RESOURCE_OPTIONS)
        actor.cargo[resource] = rng.uniform(0.5, 3.5)
        chosen = make_action("unload", resource=resource, amount=2)
    elif action_name == "transfer_food":
        actor.role, actor.cell = "forager", center
        _quiet(world, [center, *world.cells[center].neighbors], rng)
        recipient = world.bees["bee-002"]
        recipient.cell, recipient.role = center, "storer"
        recipient.cargo = dict.fromkeys(recipient.cargo, 0.0)
        resource = rng.choice(RESOURCE_OPTIONS)
        actor.cargo[resource] = rng.uniform(1.0, 3.5)
        chosen = make_action("transfer_food", target=recipient.id,
                             resource=resource, amount=1)
    elif action_name == "feed_brood":
        actor.role, actor.cell = "nurse", center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        brood = _put_brood(world, center, rng, rng.uniform(0.55, 0.9))
        world.cells[center].stores["honey"] = rng.uniform(2, 8)
        world.cells[center].stores["pollen"] = rng.uniform(1, 5)
        chosen = make_action("feed_brood", target=brood.id, amount=2)
    elif action_name == "clean":
        actor.role, actor.cell = "cleaner", center
        _quiet(world, [center, *world.cells[center].neighbors], rng)
        cell = world.cells[center]
        cell.debris = rng.uniform(2.0, 8.0)
        cell.pathogen = rng.uniform(1.0, 4.0)
        chosen = make_action("clean", amount=2)
    elif action_name == "groom":
        actor.role, actor.cell = "nurse", center
        _quiet(world, [center, *world.cells[center].neighbors], rng)
        target = world.bees["bee-002"]
        target.cell = center
        cell = world.cells[center]
        cell.pathogen = rng.uniform(2.0, 6.0)
        cell.debris = rng.uniform(0.05, 0.25)
        chosen = make_action("groom", target=target.id, amount=2)
    elif action_name == "build_comb":
        actor.role, actor.cell = "builder", center
        _quiet(world, [center, *world.cells[center].neighbors], rng)
        world.cells[center].structure = rng.uniform(0.2, 0.62)
        world.cells[center].stores["honey"] = rng.uniform(3, 10)
        chosen = make_action("build_comb", amount=2)
    elif action_name == "heat":
        actor.role, actor.cell = "thermoregulator", center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        world.cells[center].temperature_c = rng.uniform(29.5, 32.0)
        _put_brood(world, center, rng, 0.25)
        chosen = make_action("heat", amount=rng.choice((1.0, 1.5, 2.0)))
    elif action_name == "fan":
        actor.role, actor.cell = "thermoregulator", center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        world.cells[center].temperature_c = rng.uniform(38, 40)
        _put_brood(world, center, rng, 0.25)
        chosen = make_action("fan", amount=rng.choice((1.0, 1.5, 2.0)))
    elif action_name == "evaporative_cool":
        actor.role, actor.cell = "thermoregulator", center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        world.cells[center].temperature_c = rng.uniform(38, 40)
        _put_brood(world, center, rng, 0.25)
        if rng.random() < 0.5:
            actor.cargo["water"] = rng.uniform(0.8, 2)
        else:
            world.cells[center].stores["water"] = rng.uniform(0.8, 2)
        chosen = make_action("evaporative_cool", amount=1)
    elif action_name == "communicate":
        actor.role, actor.cell = "forager", "field:gate"
        recipient = world.bees["bee-002"]
        recipient.cell, recipient.role = actor.cell, "forager"
        meadow, orchard = world.cells["field:meadow"], world.cells["field:orchard"]
        meadow.flowers["nectar"] = rng.uniform(0, 1)
        orchard.flowers["nectar"] = rng.uniform(4, 18)
        message = "Meadow nectar is scarce; the nearby orchard patch has nectar."
        chosen = make_action("communicate", target=recipient.id, message=message, amount=1)
    elif action_name == "lay_eggs":
        actor.role, actor.cell = "queen", center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        # Keep fewer than four living larvae in the queen's current compartment.
        for brood in world.brood.values():
            brood.cell = "h:-1:0"
        world.cells[center].stores["honey"] = rng.uniform(2, 8)
        chosen = make_action("lay_eggs", amount=1)
    elif action_name == "switch_role":
        actor.cell = center
        local = [center, *world.cells[center].neighbors]
        _quiet(world, local, rng)
        if index % 2:
            # An unassigned bee adopts a task preference that fits a visible need.
            actor.role = "unassigned"
            world.cells[center].debris = rng.uniform(2.5, 5)
            destination = "cleaner"
        else:
            # An existing task preference steps aside for an incompatible urgent task.
            actor.role = "forager"
            _put_brood(world, center, rng, 0.75)
            world.cells[center].stores["honey"] = rng.uniform(2, 7)
            world.cells[center].stores["pollen"] = rng.uniform(1, 5)
            destination = "unassigned"
        chosen = make_action("switch_role", target=destination, amount=1)
    elif action_name == "guard":
        actor.role, actor.cell = "guard", entrance
        _quiet(world, [entrance, *world.cells[entrance].neighbors], rng)
        world.cells[entrance].debris = rng.uniform(0.4, 2)
        world.cells[entrance].pathogen = rng.uniform(1.5, 5)
        chosen = make_action("guard", amount=1)
    else:
        raise ValueError(f"Unhandled action family {action_name}")

    if chosen is None:
        raise AssertionError(action_name)
    error = validate_action(chosen)
    if error:
        raise ValueError(f"Invalid generated action {action_name}: {error}")
    observation = world.observe(actor.id)
    # A training answer must be physically applicable in the exact state shown.
    outcome = world.execute(actor.id, chosen, 15)
    if outcome["status"] != "applied":
        raise ValueError(f"Unexecutable label {action_name}: {outcome}")

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(observation, ensure_ascii=False,
                                                separators=(",", ":"), allow_nan=False)},
    ]
    target = dict(chosen)
    target["memory"] = MEMORY[action_name]
    completion = [{"role": "assistant", "content": json.dumps(
        target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}]
    row = {"prompt": prompt, "completion": completion}
    return row


def build_dataset(output_dir: str | Path, examples_per_action: int = 700,
                  seed: int = 20261102) -> dict:
    if examples_per_action < 10:
        raise ValueError("Use at least 10 examples per action for all three splits.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    class_counts = {split: Counter() for split in splits}
    master = random.Random(seed)
    for action_index, action_name in enumerate(ACTIONS_ORDER):
        records = []
        for index in range(examples_per_action):
            example_seed = master.randrange(1, 2**31-1)
            rng = random.Random(example_seed)
            records.append(_make_scenario(action_name, rng, example_seed, index))
        master.shuffle(records)
        n_test = max(1, round(examples_per_action*0.10))
        n_val = max(1, round(examples_per_action*0.10))
        n_train = examples_per_action - n_test - n_val
        if n_train < 1:
            raise ValueError("Not enough examples for three partitions.")
        partitions = {
            "train": records[:n_train],
            "validation": records[n_train:n_train+n_val],
            "test": records[n_train+n_val:],
        }
        for split, rows in partitions.items():
            splits[split].extend(rows)
            class_counts[split][action_name] += len(rows)
    for split, rows in splits.items():
        random.Random(seed+{"train": 101, "validation": 202, "test": 303}[split]).shuffle(rows)
        path = output_dir/f"bee_decisions_{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False)+"\n")
    manifest = {
        "dataset_name": "bee_colony_local_action_sft_synthetic",
        "dataset_version": VERSION,
        "seed": seed,
        "examples_per_action": examples_per_action,
        "action_names": list(ACTIONS_ORDER),
        "rows_per_split": {key: len(value) for key, value in splits.items()},
        "class_counts": {key: dict(value) for key, value in class_counts.items()},
        "format": "TRL conversational prompt-completion JSONL",
        "input_contract": "system prompt + exact local observation JSON from beehive_sim.World.observe",
        "output_contract": "one six-field action JSON object from beehive_sim.ACTION_SCHEMA",
        "label_origin": "deterministic scenario teacher and executable action preconditions",
        "limitations": [
            "Synthetic labels are operationally plausible demonstrations, not expert biological ground truth.",
            "The teacher deliberately balances action classes; it is not a natural frequency distribution.",
            "The held-out partitions use new seeded draws from the same scenario families; they are not independent biological validation.",
            "Review and replace labels with domain-expert or trusted teacher decisions before making scientific claims.",
        ],
        "sha256": {},
    }
    for split in splits:
        path = output_dir/f"bee_decisions_{split}.jsonl"
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    card = """# Bee decision SFT dataset

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
"""
    (output_dir/"DATASET_CARD.md").write_text(card, encoding="utf-8")
    return manifest


def make_zip(dataset_dir: str | Path, archive_path: str | Path):
    dataset_dir, archive_path = Path(dataset_dir), Path(archive_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(dataset_dir.iterdir()):
            if path.is_file() and (path.suffix == ".jsonl" or path.name in
                                   ("manifest.json", "DATASET_CARD.md")):
                archive.write(path, arcname=path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/bee_decision_sft")
    parser.add_argument("--examples-per-action", type=int, default=700)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--zip", default="bee_decision_sft_dataset.zip")
    args = parser.parse_args()
    manifest = build_dataset(args.output_dir, args.examples_per_action, args.seed)
    make_zip(args.output_dir, args.zip)
    print(json.dumps({"archive": args.zip, **manifest}, indent=2))


if __name__ == "__main__":
    main()
