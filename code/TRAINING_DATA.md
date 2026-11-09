# Bee decision SFT dataset: data sheet and training notes

## What's included

The generated archive bee_decision_sft_dataset.zip contains three UTF-8 JSONL files:

- bee_decisions_train.jsonl: 9,520 examples.
- bee_decisions_validation.jsonl: 1,190 examples.
- bee_decisions_test.jsonl: 1,190 examples.
- manifest.json: seed, per-action counts, hashes, format and label provenance.
- DATASET_CARD.md: short reuse/interpretation caveat.

Every line is one independent, single-decision conversation, not a multi-turn hive transcript:

~~~json
{
  "prompt": [
    {"role": "system", "content": "the bee system instruction"},
    {"role": "user", "content": "{\"simulated_time\":\"...\",\"self\":{...}, ...}"}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"action\":\"feed_brood\", ...}"}
  ]
}
~~~

The user content follows the current beehive_sim.World.observe() field schema,
including self status, location, local and adjacent physical readings, brood,
co-located bees, weather, personal memory, previous outcome, local inbox and
action guide. It contains no observer vector or global hive inventory.

The assistant completion has exactly the simulator's six fields:
action, target, resource, amount, message and memory. Labels are validated
with the action schema, then World.execute() must return applied for the same
observation state before the pair is saved.

All 17 actions are balanced: 700 examples for each class, partitioned per class
into 560 train, 70 validation and 70 test examples. That yields 9,520 / 1,190 /
1,190 total. Deliberate balancing helps each action class appear during SFT; it
does not estimate real colony action frequency.

## How labels are produced

generate_bee_sft_dataset.py builds randomized local state scenarios with the
actual simulator world and observer, then applies transparent recipes:

- hungry adult with accessible honey/nectar → eat;
- field resource available and crop not full → forage;
- loaded forager at a return location → move toward the nest;
- local receiver with crop room → transfer food; otherwise store cargo;
- hungry visible larva with sugar and pollen → feed;
- elevated debris → clean; contamination with a nearby bee and little debris → groom;
- weak comb with honey → build; cold/hot brood compartment → heat/fan/cool;
- local patch information for a visible nearby forager → communicate;
- queen plus brood cell space and honey → lay one egg;
- local task/caste/role checks → switch role or guard;
- no urgent locally observable action → rest.

The generator introduces seeded variation in positions, ages, resource amounts,
hunger, temperature, humidity, contamination, structure, role, energy, health,
memories, recent outcomes and weather time. No state vector or global reserve
score is provided to the hypothetical bee.

## Important quality limit

These labels are synthetic heuristics, not expert annotation or a reference
optimal policy. Executable means the engine accepts the action's local physical
preconditions; it does not mean an entomologist would choose that action, that
it improves overall colony welfare, or that the simplified physics are valid.
For example, role switching, grooming, feeding rates and reserve scarcity are
coarse abstractions.

The dataset deliberately repeats task families with new random seeds. A
stratified random split preserves all action classes, so held-out scores measure
interpolation across instances of these same recipes. They are not evidence of
generalization to new biology, weather regimes, or colonies. The small Qwen
may memorize cues and JSON shape without learning durable social planning.

Before a scientific deployment:

1. Have a bee-domain expert review and correct the scenario labels.
2. Add hand-annotated or independently validated teacher traces, with provenance.
3. Hold out complete scenario families and real conditions, not random rows only.
4. Evaluate exact JSON validity, engine precondition success, local interactions
   and hive-vector trajectories across multiple seeds.
5. Compare against the script teacher and original LLM on matched initial states.
   Treat vector improvement as a measured result, not a training label.

## Training recipe

train_qwen35_bee_agents.ipynb loads prompt-completion rows with TRL's
completion-only objective. It uses Unsloth and unsloth/Qwen3.5-0.8B, 16-bit base
weights and trainable rank-16 LoRA adapters. The notebook compares base and
fine-tuned generations on the same 51 held-out examples, then saves the adapter.
Reported metrics measure imitation/format quality relative to synthetic labels.


## Running the trained adapter in the simulator

The companion bee_local_qwen_policy.py exposes BeeLocalQwenPolicy. After training:

~~~python
from bee_local_qwen_policy import BeeLocalQwenPolicy

local_policy = BeeLocalQwenPolicy(model, tokenizer)
# local_policy implements decide(observation) and can be passed to Simulation.
action = local_policy.decide(world.observe("bee-001"))
~~~

For a later runtime, call BeeLocalQwenPolicy.from_adapter("/content/bee_qwen35_08b_bee_lora/adapter").
It uses the local checkpoint and Unsloth only; it does not call an API. The simulator
still enforces movement, crop, brood, caste and energy preconditions after the
policy returns its strict JSON object.
