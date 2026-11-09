import csv
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from beehive_sim import (
    ACTION_SCHEMA, VECTOR_NAMES, Config, OfflineDemoPolicy, Observer, OpenAIPolicy,
    PolicyError, Simulation, World, make_action, validate_action, weighted_research_score,
)


class RestPolicy:
    label = "test_rest"
    def __init__(self):
        self.observations = []
    def decide(self, observation):
        self.observations.append(observation)
        return make_action("rest", memory=observation["self"]["id"])


def config(**overrides):
    return Config(**({"n_bees": 8, "duration_hours": 0.5,
                       "round_minutes": 15, "max_actions": 100} | overrides))


def total_resource(world, resource):
    return (sum(c.stores[resource] for c in world.cells.values()) +
            sum(c.flowers.get(resource, 0) for c in world.cells.values()) +
            (sum(c.water_source for c in world.cells.values()) if resource == "water" else 0) +
            sum(b.cargo[resource] for b in world.bees.values()))


def test_fifty_default_bees_and_adjustability():
    assert Config().n_bees == 50
    for n in (2, 13, 50):
        world = World(config(n_bees=n))
        sim = Simulation(world, RestPolicy())
        rows = sim.run()
        assert len(world.bees) == n
        assert len(rows) == n*2
        assert world.minutes == pytest.approx(30)
        assert len(sim.audit) == len(rows)
        assert len({e["bee_id"] for e in sim.audit[:n]}) == n
        timestamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
        assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
        assert all(len(r["vector"]) == len(VECTOR_NAMES) for r in rows)
        assert all(math.isfinite(v) and 0 <= v <= 1 for r in rows for v in r["vector"])


def test_observer_anchors_orientation_and_dead_cohort_penalty():
    world = World(config())
    observer = Observer(world)
    before = dict(zip(VECTOR_NAMES, observer.snapshot()))
    for key in world.hive_cells:
        cell = world.cells[key]
        cell.debris = 1000
        cell.pathogen = 1000
        cell.structure = 0
        cell.temperature_c = 90
        cell.humidity_pct = 0
        cell.stores = dict.fromkeys(cell.stores, 0)
    for bee in world.bees.values():
        bee.health = bee.energy = 0
        bee.alive = False
    for brood in world.brood.values():
        brood.alive = False
    after = dict(zip(VECTOR_NAMES, observer.snapshot()))
    assert all(0 <= v <= 1 for v in after.values())
    for name in VECTOR_NAMES:
        assert after[name] == 0
        assert before[name] > after[name]
    world.cells["h:0:0"].debris = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        observer.snapshot()


def test_hidden_distant_changes_do_not_change_agent_observation():
    world = World(config())
    actor = world.bees["bee-001"]
    actor.cell = "h:-2:0"
    first = world.observe(actor.id)
    hidden = world.cells["h:2:0"]  # not current or adjacent
    hidden.stores["honey"] += 1000
    hidden.debris = 100
    hidden.pathogen = 5
    second = world.observe(actor.id)
    assert first == second
    assert set(first) == {"simulated_time", "self", "current_cell", "neighbor_cells",
                          "co_located_bees", "weather", "personal_memory",
                          "last_action_outcome", "received_messages", "action_guide"}
    assert not any(name in json.dumps(first) for name in VECTOR_NAMES)
    assert "vector" not in json.dumps(first)
    first["self"]["cargo_units"]["honey"] = 999
    assert actor.cargo["honey"] != 999


def test_peer_messages_and_memories_remain_local():
    world = World(config())
    a, b, c = (world.bees[f"bee-{i:03d}"] for i in (1, 2, 3))
    a.cell = b.cell = "h:-2:0"
    c.cell = "h:2:0"
    a.memory = "a private memory"
    result = world.execute(a.id, make_action("communicate", message="Need a receiver"), 15)
    assert result["status"] == "applied"
    assert world.observe(b.id)["received_messages"][0]["message"] == "Need a receiver"
    assert world.observe(c.id)["received_messages"] == []
    assert world.observe(b.id)["personal_memory"] == ""
    assert "a private memory" not in json.dumps(world.observe(b.id))
    world.advance(world.config.message_ttl_minutes+1)
    assert not world.observe(b.id)["received_messages"]


@pytest.mark.parametrize("resource", ["nectar", "pollen", "water"])
def test_forage_transfer_unload_conserve_mass(resource):
    world = World(config())
    a, b = world.bees["bee-001"], world.bees["bee-002"]
    patch = "field:pond" if resource == "water" else "field:meadow"
    a.cell = b.cell = patch
    world.minutes = 60  # daylight
    before = total_resource(world, resource)
    outcome = world.execute(a.id, make_action("forage", resource=resource), 15)
    assert outcome["status"] == "applied"
    assert total_resource(world, resource) == pytest.approx(before)
    outcome = world.execute(a.id, make_action("transfer_food", target=b.id, resource=resource), 15)
    assert outcome["status"] == "applied"
    assert total_resource(world, resource) == pytest.approx(before)
    b.cell = "h:2:0"  # position setup, not movement tested here
    outcome = world.execute(b.id, make_action("unload", resource=resource), 15)
    assert outcome["status"] == "applied"
    assert total_resource(world, resource) == pytest.approx(before)


def test_physical_constraints_reject_impossible_actions_without_resource_change():
    world = World(config())
    bee = world.bees["bee-001"]
    bee.cell = "h:-2:0"
    before = json.dumps(world.raw_state(), sort_keys=True)
    assert world.execute(bee.id, make_action("move", target="field:meadow"), 15)["status"] == "rejected"
    assert world.execute(bee.id, make_action("lay_eggs"), 15)["status"] == "rejected"
    assert world.execute(bee.id, make_action("switch_role", target="forager"), 15)["status"] == "rejected"
    assert json.dumps(world.raw_state(), sort_keys=True) == before
    assert world.execute(bee.id, make_action("switch_role", target="unassigned"), 15)["status"] == "applied"
    assert world.execute(bee.id, make_action("switch_role", target="forager"), 15)["status"] == "applied"
    queen = world.bees["bee-000"]
    queen.cell = "h:2:0"
    assert world.execute(queen.id, make_action("move", target="field:gate"), 15)["status"] == "rejected"


def test_brood_care_consumes_stores_and_egg_laying_does_not_improve_cohort_score():
    world = World(config())
    bee = world.bees["bee-001"]
    brood = next(iter(world.brood.values()))
    bee.cell = brood.cell
    hunger_before, sugar_before = brood.hunger, total_resource(world, "honey")
    pollen_before = total_resource(world, "pollen")
    outcome = world.execute(bee.id, make_action("feed_brood", target=brood.id), 15)
    assert outcome["status"] == "applied"
    assert brood.hunger < hunger_before
    assert total_resource(world, "honey") < sugar_before
    assert total_resource(world, "pollen") < pollen_before
    observer = Observer(world)
    before = observer.snapshot()[6:8]
    queen = world.bees["bee-000"]
    queen.cell = "h:0:0"
    assert world.execute(queen.id, make_action("lay_eggs"), 15)["status"] == "applied"
    assert observer.snapshot()[6:8] == before
    egg = list(world.brood.values())[-1]
    assert egg.stage == "egg" and not egg.initial


def test_thermal_actions_have_context_dependent_direction_and_energy_cost():
    world = World(config())
    bee = world.bees["bee-001"]
    bee.cell = "h:0:0"
    cell = world.cells[bee.cell]
    observer = Observer(world)
    cell.temperature_c = 33
    before, energy = observer.snapshot()[8], bee.energy
    world.execute(bee.id, make_action("heat"), 15)
    assert observer.snapshot()[8] > before
    assert bee.energy < energy
    cell.temperature_c = 36
    before = observer.snapshot()[8]
    world.execute(bee.id, make_action("heat"), 15)
    assert observer.snapshot()[8] < before


def test_background_and_immediate_action_deltas_telescope():
    sim = Simulation(World(config()), OfflineDemoPolicy())
    sim.run()
    for index in range(len(VECTOR_NAMES)):
        delta_sum = sum(e["immediate_action_delta"][index]+e["background_delta"][index]
                        for e in sim.audit)
        assert delta_sum == pytest.approx(sim.transcript[-1]["vector"][index]-sim.baseline[index], abs=1e-7)
    assert any(any(abs(v) > 0 for v in e["background_delta"]) for e in sim.audit)


def test_rejected_json_proposal_gets_exactly_one_snapshot():
    class InvalidPolicy:
        label = "invalid_test"
        def decide(self, observation):
            return {"action": "teleport"}
    sim = Simulation(World(config(max_actions=3)), InvalidPolicy())
    assert len(sim.run()) == 3
    assert all(e["outcome"]["status"] == "rejected" for e in sim.audit)
    assert all(not any(e["immediate_action_delta"]) for e in sim.audit)


def test_partial_outputs_preserved_without_fallback_on_policy_failure(tmp_path):
    class FailingPolicy(RestPolicy):
        label = "failing_test"
        def decide(self, observation):
            if self.observations:
                raise PolicyError("simulated transport failure")
            return super().decide(observation)
    out = tmp_path/"partial"
    sim = Simulation(World(config()), FailingPolicy(), out)
    with pytest.raises(PolicyError):
        sim.run()
    assert len(sim.transcript) == len(sim.audit) == 1
    assert len((out/"vectors.jsonl").read_text().splitlines()) == 1
    metadata = json.loads((out/"metadata.json").read_text())
    assert not metadata["completed"]
    assert metadata["events_recorded"] == 1
    assert metadata["stop_reason"] == "interrupted:PolicyError"
    assert (out/"vectors.csv").exists()


def test_jsonl_csv_and_metadata_align_and_existing_runs_are_not_overwritten(tmp_path):
    out = tmp_path/"run"
    sim = Simulation(World(config(max_actions=11)), OfflineDemoPolicy(), out)
    transcript = sim.run()
    assert [json.loads(line) for line in (out/"vectors.jsonl").read_text().splitlines()] == transcript
    assert len((out/"actions.jsonl").read_text().splitlines()) == 11
    with (out/"vectors.csv").open() as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["timestamp", *VECTOR_NAMES]
    assert len(rows) == 12
    metadata = json.loads((out/"metadata.json").read_text())
    assert metadata["events_recorded"] == 11
    assert metadata["completed"] and metadata["stop_reason"] == "max_actions"
    assert metadata["policy"] == "offline_demo_NOT_LLM"
    with pytest.raises(FileExistsError):
        Simulation(World(config()), RestPolicy(), out).run()


def test_offline_reproducibility_and_private_memories():
    first = Simulation(World(config()), OfflineDemoPolicy(123))
    second = Simulation(World(config()), OfflineDemoPolicy(123))
    assert first.run() == second.run()
    policy = RestPolicy()
    sim = Simulation(World(config()), policy)
    sim.run()
    n = sim.world.config.n_bees
    assert all(o["personal_memory"] == "" for o in policy.observations[:n])
    assert all(o["personal_memory"] == o["self"]["id"] for o in policy.observations[n:])


def test_openai_request_uses_requested_model_schema_and_only_local_observation():
    calls = []
    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="fake-response", model="gpt-6-luna",
                                   status="completed", output_text=json.dumps(make_action("rest")),
                                   usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 12, "output_tokens": 6}))
    cfg = config()
    policy = OpenAIPolicy(cfg, client=SimpleNamespace(responses=Responses()))
    observation = World(cfg).observe("bee-001")
    assert policy.decide(observation)["action"] == "rest"
    call = calls[0]
    assert call["model"] == "gpt-6-luna"
    assert json.loads(call["input"]) == observation
    assert call["text"]["format"]["schema"] == ACTION_SCHEMA
    assert call["text"]["format"]["strict"] is True
    assert call["store"] is False
    assert "previous_response_id" not in call
    assert policy.last_usage["input_tokens"] == 12


def test_openai_incomplete_or_non_json_response_never_generates_an_action():
    class Responses:
        def create(self, **kwargs):
            return SimpleNamespace(id="fake", model="gpt-6-luna", status="incomplete", usage=None)
    policy = OpenAIPolicy(config(), client=SimpleNamespace(responses=Responses()))
    with pytest.raises(PolicyError, match="no action"):
        policy.decide({})


@pytest.mark.parametrize("kwargs", [
    {"n_bees": 1}, {"n_bees": 3.5}, {"duration_hours": 0},
    {"reserve_hours": float("nan")}, {"round_minutes": -1},
    {"start_timestamp": "2026-01-01T00:00:00"}, {"crop_capacity": 0},
])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_weighted_analysis_requires_explicit_valid_weights():
    assert weighted_research_score([0.5]*14, [1]*14) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        weighted_research_score([0.5]*14, [0]*14)
    with pytest.raises(ValueError):
        weighted_research_score([0.5]*14, [-1]+[1]*13)
    assert validate_action(make_action("heat", amount=float("nan"))) is not None


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", None])
def test_incomplete_reason_is_reported_and_saved_without_executing_partial_action(tmp_path, reason):
    calls = []
    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                id="incomplete-response", model="gpt-6-luna", status="incomplete",
                incomplete_details=SimpleNamespace(reason=reason),
                output_text=json.dumps(make_action("rest")),
                usage=SimpleNamespace(model_dump=lambda **kw: {"input_tokens": 100, "output_tokens": 550}),
            )
    policy = OpenAIPolicy(config(), client=SimpleNamespace(responses=Responses()))
    out = tmp_path/"incomplete"
    sim = Simulation(World(config()), policy, out)
    with pytest.raises(PolicyError) as caught:
        sim.run()
    assert f"reason: {reason or 'not supplied'}" in str(caught.value)
    assert ("increase config.max_output_tokens" in str(caught.value)) == (reason == "max_output_tokens")
    assert len(calls) == 1
    assert sim.transcript == [] and sim.audit == []
    metadata = json.loads((out/"metadata.json").read_text())
    assert metadata["last_policy_response"]["status"] == "incomplete"
    assert metadata["last_policy_response"]["incomplete_reason"] == reason
    assert metadata["last_policy_response"]["usage"]["output_tokens"] == 550
    assert metadata["last_policy_response"]["response_id"] == "incomplete-response"
    assert not metadata["completed"]


def test_incomplete_diagnostics_reset_when_next_call_completes():
    calls = []
    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                id=f"response-{len(calls)}", model="gpt-6-luna",
                status="incomplete" if len(calls) == 1 else "completed",
                incomplete_details={"reason": "max_output_tokens"} if len(calls) == 1 else None,
                output_text=json.dumps(make_action("rest")), usage=None,
            )
    policy = OpenAIPolicy(config(), client=SimpleNamespace(responses=Responses()))
    with pytest.raises(PolicyError):
        policy.decide({})
    assert policy.last_incomplete_reason == "max_output_tokens"
    assert policy.decide({})["action"] == "rest"
    assert policy.last_incomplete_reason is None
    assert policy.last_response_status == "completed"
    assert calls[-1]["max_output_tokens"] == 2000
    assert calls[-1]["reasoning"] == {"effort": "none"}


def response_message(text, phase=None, content=None, status="completed"):
    return SimpleNamespace(type="message", role="assistant", phase=phase,
                           status=status, id="msg-test", content=content if content is not None else
                           [SimpleNamespace(type="output_text", text=text)])


def completed_response(output, aggregate=""):
    return SimpleNamespace(id="response-output-test", model="gpt-6-luna",
                           status="completed", output=output, output_text=aggregate, usage=None)


def policy_for_response(response):
    return OpenAIPolicy(config(), client=SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: response)))


def test_action_uses_final_message_instead_of_concatenated_commentary():
    intended = make_action("rest")
    other = json.dumps(make_action("move", target="h:0:0"))
    response = completed_response(
        [response_message(other, "commentary"),
         SimpleNamespace(type="reasoning", summary=[]),
         response_message(json.dumps(intended), "final_answer")],
        other+"\n"+json.dumps(intended),
    )
    policy = policy_for_response(response)
    sim = Simulation(World(config(max_actions=1)), policy)
    assert len(sim.run()) == 1
    assert sim.audit[0]["action"] == intended
    assert policy.last_output_diagnostics is None


def test_action_reads_unlabelled_message_and_split_content_parts():
    intended = make_action("rest", memory="literal braces: } {")
    raw = json.dumps(intended)
    content = [SimpleNamespace(type="output_text", text=raw[:30]),
               SimpleNamespace(type="output_text", text=raw[30:])]
    response = completed_response([response_message("", content=content)])
    assert policy_for_response(response).decide({}) == intended


@pytest.mark.parametrize("output", [
    [response_message(json.dumps(make_action("rest"))),
     response_message(json.dumps(make_action("heat")))],
    [response_message(json.dumps(make_action("rest")), "final_answer"),
     response_message(json.dumps(make_action("heat")), "final_answer")],
    [response_message("Thinking", "commentary")],
    [],
])
def test_ambiguous_or_missing_final_messages_do_not_create_actions(tmp_path, output):
    policy = policy_for_response(completed_response(output))
    sim = Simulation(World(config()), policy, tmp_path/"ambiguous")
    with pytest.raises(PolicyError, match="final action message"):
        sim.run()
    assert sim.transcript == []
    assert policy.last_output_diagnostics is not None


@pytest.mark.parametrize("raw", [
    json.dumps(make_action("rest"))+"\n"+json.dumps(make_action("heat")),
    json.dumps(make_action("rest"))+"\nextra explanation",
    chr(96)*3+"json\n"+json.dumps(make_action("rest"))+"\n"+chr(96)*3,
])
def test_single_message_extra_or_non_json_text_is_not_silently_truncated(tmp_path, raw):
    policy = policy_for_response(completed_response([response_message(raw)], raw))
    sim = Simulation(World(config()), policy, tmp_path/"bad-json")
    with pytest.raises(PolicyError, match="Action JSON is invalid"):
        sim.run()
    assert sim.transcript == []
    metadata = json.loads((tmp_path/"bad-json"/"metadata.json").read_text())
    diagnostics = metadata["last_policy_response"]["output_diagnostics"]
    assert diagnostics["aggregate_output_text"] == raw
    assert diagnostics["messages"][0]["content"][0]["text"] == raw


def test_refusal_or_incomplete_final_message_is_not_executed():
    refusal = response_message("", content=[SimpleNamespace(type="refusal", refusal="Declined")])
    with pytest.raises(PolicyError, match="refused"):
        policy_for_response(completed_response([refusal])).decide({})
    incomplete = response_message(json.dumps(make_action("rest")), status="incomplete")
    with pytest.raises(PolicyError, match="incomplete"):
        policy_for_response(completed_response([incomplete])).decide({})
