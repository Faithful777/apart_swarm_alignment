"""Local, deterministic action menu for the Qwen bee policy.

Uses only World.observe fields. No observer vector or global hive state enters the menu.
An action can still fail at execution if an earlier bee changes the frozen batch state.
"""
from __future__ import annotations

from typing import Any
from beehive_sim import make_action, validate_action

_ACTION_KEYS = ("action", "target", "resource", "amount", "message")
_COST = {"move": 1.2, "rest": 0.0, "eat": 0.1, "forage": 2.0,
         "unload": 0.2, "transfer_food": 0.2, "feed_brood": 0.4,
         "clean": 0.8, "groom": 0.5, "build_comb": 1.4,
         "heat": 2.0, "fan": 1.2, "evaporative_cool": 0.8,
         "communicate": 0.1, "lay_eggs": 1.0, "guard": 0.8}


def physical_signature(action: dict[str, Any]) -> tuple:
    return tuple(action.get(key) for key in _ACTION_KEYS)


def feasible_actions(observation: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """Return bounded locally feasible proposals, including a guaranteed rest choice."""
    if limit < 2:
        raise ValueError("limit must be at least two")
    bee = observation["self"]
    cell = observation["current_cell"]
    food = cell["food_units"]
    cargo = bee["cargo_units"]
    zone = cell["zone"]
    energy = float(bee["energy_percent"])
    current_id = cell["id"]
    chosen: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def add(name: str, target: str = "", resource: str = "none",
            amount: float = 1.0, message: str = ""):
        movement_cost = 2.0 if name == "move" and (
            zone == "outside" or any(c["id"] == target and c["zone"] == "outside"
                                     for c in observation["neighbor_cells"])) else 1.0
        if energy + 1e-6 < _COST[name] * movement_cost:
            return
        action = make_action(name, target=target, resource=resource, amount=amount,
                             message=message, memory=f"I chose {name} at {current_id}.")
        if validate_action(action) is not None:
            raise AssertionError(f"Invalid menu action: {action}")
        signature = physical_signature(action)
        if signature not in seen:
            seen.add(signature)
            chosen.append(action)

    # Local needs and tasks precede travel. The menu is intentionally small.
    if energy < 99.9:
        for resource in ("honey", "nectar"):
            if float(cargo[resource]) + (float(food[resource]) if zone != "outside" else 0.0) > 1e-5:
                add("eat", resource=resource, amount=2)
    if zone == "outside":
        room = float(bee["crop_capacity"]) - sum(float(x) for x in cargo.values())
        if room > 1e-5:
            if float(cell["water_source_units"]) > 1e-5:
                add("forage", resource="water", amount=2)
            if observation["weather"]["daylight"] > 0:
                for resource in ("nectar", "pollen"):
                    if float(cell["flower_units"][resource]) > 1e-5:
                        add("forage", resource=resource, amount=2)
    else:
        for resource in ("nectar", "pollen", "water", "honey"):
            if float(cargo[resource]) > 1e-5:
                add("unload", resource=resource, amount=2)
        larvae = [b for b in cell["brood"] if b["stage"] == "larva"
                  and float(b["hunger_percent"]) > 0.01]
        sugar = sum(float(cargo[r]) + float(food[r]) for r in ("honey", "nectar"))
        pollen = float(cargo["pollen"]) + float(food["pollen"])
        if larvae and sugar > 1e-5 and pollen > 1e-5:
            add("feed_brood", target=max(larvae, key=lambda b: b["hunger_percent"])["id"], amount=2)
        if float(cell["debris_units"]) > 1e-5:
            add("clean", amount=2)
        if float(cell["comb_condition"]) < 99.99 and float(cargo["honey"]) + float(food["honey"]) > 1e-5:
            add("build_comb", amount=2)
        if float(cell["temperature_c"]) < 34.5:
            add("heat", amount=1)
        if float(cell["temperature_c"]) > 35.5:
            add("fan", amount=1)
            if float(cargo["water"]) + float(food["water"]) > 1e-5:
                add("evaporative_cool", amount=1)
        if float(cell["contamination_units"]) > 1e-5:
            add("groom", amount=1)
        if bee["caste"] == "queen" and zone == "brood" and len(cell["brood"]) < 4 and (
                float(cargo["honey"]) + float(food["honey"])) >= 0.5:
            add("lay_eggs", amount=1)
        if zone == "entrance":
            add("guard", amount=1)
    if observation["co_located_bees"]:
        add("communicate", amount=1,
            message=f"I am at {current_id}; nearby work may need help.")
        room = float(bee["crop_capacity"])
        for peer in observation["co_located_bees"][:2]:
            peer_room = room - sum(float(x) for x in peer["cargo_units"].values())
            if peer_room > 1e-5:
                for resource in ("honey", "nectar", "pollen", "water"):
                    if float(cargo[resource]) > 1e-5:
                        add("transfer_food", target=peer["id"], resource=resource, amount=1)
                        break
    neighbors = [c for c in observation["neighbor_cells"]
                 if not (bee["caste"] == "queen" and c["zone"] == "outside")]
    if bee["role"] == "forager":
        neighbors.sort(key=lambda c: (c["zone"] != "outside", c["id"]))
    elif bee["role"] == "nurse":
        neighbors.sort(key=lambda c: (c["zone"] != "brood", c["id"]))
    else:
        neighbors.sort(key=lambda c: c["id"])
    for neighbor in neighbors[:3]:
        add("move", target=neighbor["id"], amount=1)
    rest = make_action("rest", amount=0, memory=f"I rested at {current_id}.")
    return chosen[:limit - 1] + [rest]


def teacher_action(observation: dict[str, Any], menu: list[dict[str, Any]],
                   original: dict[str, Any] | None = None) -> dict[str, Any]:
    """Choose a valid demonstration, retaining an applicable original proposal."""
    if original:
        same = [a for a in menu if physical_signature(a) == physical_signature(original)]
        if same:
            return same[0]
        same_task = [a for a in menu if a["action"] == original.get("action")]
        if same_task:
            return same_task[0]
    bee, cell = observation["self"], observation["current_cell"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for action in menu:
        by_name.setdefault(action["action"], []).append(action)
    if bee["energy_percent"] < 65 and "eat" in by_name:
        return by_name["eat"][0]
    if bee["caste"] == "queen" and "lay_eggs" in by_name:
        return by_name["lay_eggs"][0]
    if cell["zone"] == "outside" and "forage" in by_name:
        return by_name["forage"][0]
    if "unload" in by_name:
        return by_name["unload"][0]
    if "feed_brood" in by_name and bee["role"] == "nurse":
        return by_name["feed_brood"][0]
    if bee["role"] == "thermoregulator":
        for task in ("heat", "evaporative_cool", "fan"):
            if task in by_name:
                return by_name[task][0]
    if bee["role"] == "cleaner" and "clean" in by_name:
        return by_name["clean"][0]
    if bee["role"] == "builder" and "build_comb" in by_name:
        return by_name["build_comb"][0]
    if bee["role"] == "guard" and "guard" in by_name:
        return by_name["guard"][0]
    if "move" in by_name:
        return by_name["move"][0]
    return menu[-1]
