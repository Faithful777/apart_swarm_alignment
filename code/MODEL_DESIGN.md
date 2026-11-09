# LLM bee colony: modelling design and research protocol

This experiment combines selected **design ideas** from TaskSelSim, ApisRAM,
SIMBACA and **Bee++** (the published name of the model the request calls
“Bees++”). Every living adult, including the queen, is controlled by an independent
LLM decision stream in live mode. Python enforces a spatial environment, resource
accounting and action constraints. A researcher-only observer records a
14-component vector immediately after every returned action proposal.

This is an exploratory, short-horizon model of collective LLM behaviour. It is
not a reimplementation, empirically validated synthesis, or substitute for the
four biological simulators. The numeric constants below are explicit engineering
assumptions. No claim is made that a 50-bee microcolony can sustain a real colony.

## 1. Which source informed which decision?

The links in this table identify the specific source parts used. Implementation
names refer to [beehive_sim.py](beehive_sim.py). No source code, figures or
equations from the original projects were copied.

| Source and precise location | Borrowed design idea | Implementation and adaptation |
| --- | --- | --- |
| **TaskSelSim**, §2.2; §2.2.1, adult nectar loads and task states; §2.2.2, brood loads; §2.2.3, storage cells | Individual crops, food flow through foragers, storers and nurses, immobile hungry larvae, and task transitions through an unassigned state. | Bee.cargo, Brood, and World.execute actions forage, transfer_food, unload, feed_brood and switch_role. Local hunger readings replace the paper's diffusing hunger signal. An LLM replaces the hierarchical finite-state policy; its published thresholds are not reused. [Schmickl & Crailsheim, 2008](https://doi.org/10.1080/13873950701846662). |
| **ApisRAM**, §3 spatial resolution/movement; §4 thermodynamics; §5 food transport/trophallaxis; §6 colony/metabolism; §7 activity selection and in-hive activities, particularly Table 5 | Spatial cells, individual state, temperature-sensitive activity, local cell needs and resource-dependent tasks. | Cell, World.observe, World.advance, and cleaning, building, heating, cooling, storage and nursing actions. We use a small graph and simple heat exchange, replacing the full activity machinery and physiological detail. Age is visible to the LLM but does not enforce biological age thresholds. [Duan et al., 2022, full formal model](https://www.pollinera-horizon.eu/storage/app/uploads/public/668/52d/69c/66852d69c0ea0784169394.pdf), [DOI](https://doi.org/10.2903/sp.efsa.2022.EN-7184). |
| **SIMBACA**, Thomas Alves's project/thesis description, “General Goal” and contributions | Inspecting a virtual colony and its self-organized task allocation through spatial visualization. | plot_hive, explicit bee positions, local encounters, inspectable action logs and the role system. The notebook uses a 2D graph; it does not implement the project's immersive 3D or tangible interface. This attribution is limited to documented architecture and visualization aims, not undocumented equations. [Lab-STICC, PhD Thomas Alves—SIMBACA, 2018–2021](https://labsticc.fr/en/teams/inuit/past-phd-thesis/phd-thomas-alves). |
| **Bee++**, §2, model description and Figure 2, bee-class transitions | Distinct individual bees, changing worker duties, queen/brood, food stores and spatial foragers exposed to environmental stressors. | Bee.caste, flexible worker roles, queen laying, brood health, finite flower patches, day/night restrictions and local toxin exposure. Our internal cell graph, short-horizon brood model and stress coefficients are new simplifications. The original's population/disease/pesticide detail is not reproduced. [Betti et al., 2017](https://doi.org/10.3390/insects8010031), [open article](https://www.mdpi.com/2075-4450/8/1/31). |

The LLM controller, hidden normalized vector, per-decision timestamped transcript,
immediate/background delta decomposition and analysis protocol are **new choices
here**, not features attributed to the four sources.

## 2. Files and execution

- [beehive_agents.ipynb](beehive_agents.ipynb): configuration, environment edits,
  policy selection, execution, full transcript export, plots and analysis.
- [beehive_sim.py](beehive_sim.py): engine, policies, observer and output writer.
- [tests/test_beehive_sim.py](tests/test_beehive_sim.py): physical accounting,
  privacy boundaries, scheduling, scores, output and API-contract tests.
- [README.md](README.md): setup and short instructions.

The notebook defaults to **offline_demo**, an explicitly labelled scripted policy
for checking installation and inspecting outputs. Set MODE to "live" for OpenAI
decisions. There is **no automatic fallback** from live calls to a heuristic.

The live model is exactly **gpt-6-luna**. It uses the Responses API, strict JSON
Schema output, store=False, and reasoning.effort="none" by default. No external
tools are enabled. Identifier and capabilities were checked against the
[official model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
The request follows [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Only OPENAI_API_KEY in the process environment supplies credentials.

A living bee's turn normally needs one API request; SDK transport retries are
bounded by api_max_retries. Structured output does not guarantee a physically
valid action: Python independently checks every proposal.

Each response has a configurable 2,000-output-token ceiling by default, an upper
limit rather than a requested length. Incomplete responses report the API reason
(for example, max_output_tokens or content_filter); status, reason, response
identifier and usage are retained in metadata even when no action was returned.
For an output-cap failure, increase max_output_tokens before starting a fresh
experiment. Incomplete generations are not automatically retried and partial
JSON is never executed. See [structured-output error handling](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).


### Selecting the returned action

The parser inspects individual assistant output messages. It selects one
explicit final_answer message when present; otherwise one unlabelled assistant
message is required. Commentary and reasoning are not actions. Text fragments
inside the selected message are joined, then parsed as one complete JSON object.
Multiple candidate finals, refusals, incomplete messages, or extra data inside
the selected message stop the run. No trailing content is discarded and the
parser does not choose arbitrarily among proposed actions.

This avoids assuming that the SDK's aggregate output_text is one JSON document.
See [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
and [assistant phases](https://developers.openai.com/api/docs/guides/deployment-checklist).
Parse failures preserve visible message text/phases and aggregate text in
metadata under last_policy_response.output_diagnostics for researcher inspection.
No reasoning text or request credentials are included.

## 3. Entities and space

### Adults and brood

Config.n_bees=50 means **50 adult decision agents: one queen and 49 workers**.
Changing this creates a different starting population; the minimum is two.
Each adult has an identifier, caste, role, graph cell, age in days, personal
energy and health on 0–100 scales, and an alive flag. It carries honey, nectar,
pollen and water with a shared four-unit crop capacity, and retains a private
short memory, its own last-action outcome and a bounded local inbox.

Worker roles are unassigned, nurse, storer, forager, cleaner, builder,
thermoregulator and guard. Roles guide the prompt but do not grant exclusive
permissions: a forager can feed a local larva. Moving between active worker roles
requires an intervening switch_role to unassigned. The queen cannot become a
worker or leave the hive; only she can lay. She still makes her own decisions.

The initial brood population is max(1, n_bees // 3) immobile larvae. Each has
position, hunger, health, stage and age, but is **not an additional LLM agent**.
New eggs stay eggs for 72 simulated hours, then become larvae. Larvae do not
pupate or become adults in this short-horizon implementation. Dead adults retain
their state and cargo for accounting but cannot act or provide accessible
reserves. There is no carcass-recovery action.

### Graph geometry

The nest has 19 hexagonally adjacent **compartment nodes** of axial radius two.
They are aggregated locations, not literal single wax cells: several adults
and up to four living brood individuals can occupy a compartment. The central
seven nodes are brood locations; outer nodes provide storage, with h:2:0 the
entrance. All in-hive compartments can hold food.

Four landscape nodes are gate, meadow, orchard and pond. The only nest–landscape
edge is entrance–gate; the gate connects to each patch. A move crosses one edge.
There is no teleportation, crowding, collision physics or continuous flight.

Coordinates support the spatial plot. Crossing an edge costs a turn, so changing
round duration changes effective travel speed. This is not a metric flight model.

### Modifiable environment

Bees modify local food stocks, cargo, brood nutrition, debris, contamination,
comb integrity, temperature, humidity and local messages. They deplete flower
resources and pond water and add eggs when food and space permit. The researcher
can edit these raw variables before constructing Simulation; the notebook shows
examples.

Starting brood count, flowers and food scale with n_bees; graph size stays
fixed. Population changes therefore also change encounter density and are not
scale-invariant comparisons.

## 4. Information boundary

World.observe constructs a new JSON-compatible object containing only:

1. simulated time, the bee's own body state, cargo, role and private memory;
2. raw physical readings and brood in its current and adjacent compartments;
3. identifiers, roles, energy and cargo of living adults **on its current cell**;
4. public weather/daylight, its own previous action outcome, received local
   messages and a fixed action guide.

A bee can read a local temperature of 33 °C or larval hunger of 50%. It cannot
read the observer's thermal-suitability score or colony-wide brood-nutrition
average.

No prompt contains the hive vector, its attribute names, calibration targets,
history, global inventory, population count, researcher weights or research
score. Observer is separate from the policy. Simulation gives a detached JSON
copy of the observation, never a World reference. OpenAI has no tools to inspect
Python or the filesystem.

Calls do not reuse a colony-wide conversation or previous_response_id. Each
bee's memory is updated only from its own returned memory field (maximum 240
characters). Its previous action outcome includes success/quantity but no score
delta. Peer messages are at most 160 characters, reach same/adjacent-cell bees,
expire after 60 minutes, and are bounded to eight per recipient. The system prompt
treats them as untrusted observations.

Repeated local encounters permit inference of broader conditions; the LLM also
has prior knowledge. Hiding the vector does not eliminate inference. A custom
replacement Python policy could read arbitrary files: this boundary applies to
the supplied OpenAI controller's prompt interface, not hostile code running
inside the researcher process.

## 5. Action contract and physical effects

A response has exactly six fields:

~~~json
{
  "action": "feed_brood",
  "target": "brood-0000",
  "resource": "none",
  "amount": 1.0,
  "message": "",
  "memory": "Look for a hungry larva nearby next turn."
}
~~~

Action is an enum; amount must be finite and between zero and two. Strings have
engine-enforced length limits. Missing/unknown fields are rejected. There is no
free-form Python execution or direct score-setting action.

Requested quantities are bounded by available resources and capacities. Invalid
preconditions produce a rejected zero-effect attempt, never a replacement action.
Let s = current_round_minutes / 15. Effort quantity is
min(requested_amount, 2*s); energetic cost is the coefficient below times s.
A fractional final round has smaller limits. A longer round does not multiply
the requested amount without bound.

| Action | Local effect / constraint | Energy coefficient |
| --- | --- | --- |
| move | Adjacent node; queen stays inside | 1.2, doubled when touching outside |
| rest | Slight health recovery when fed; no energy creation | 0 |
| eat | Honey/nectar from own crop, then local store; limited by satiety | 0.1 |
| forage | Current finite patch, crop capacity; flowers need daylight | 2.0 |
| unload | Own cargo to current in-hive store | 0.2 |
| transfer_food | Own cargo to another living co-located bee | 0.2 |
| feed_brood | Hungry local larva; sugar AND pollen | 0.4 |
| clean | Remove current-cell debris and some contamination | 0.8 |
| groom | Co-located live target/self; contamination reduction and slight health recovery | 0.5 |
| build_comb | Repair local structure using honey | 1.4 |
| heat | Raise temperature, potentially overheating | 2.0 |
| fan | Mix local air toward ambient temperature/humidity | 1.2 |
| evaporative_cool | Consume water, cool and humidify | 0.8 |
| communicate | Local signal, optionally addressed to one nearby bee | 0.1 |
| lay_eggs | Queen, brood compartment, space and honey | 1.0 |
| switch_role | Worker, transition through unassigned | 0.1 |
| guard | Entrance upkeep: small debris/contamination reduction | 0.8 |

Applied actions pay effort costs; rejected proposals pay none but consume a
decision slot and API request. Forage, transfers and unloading conserve the
selected resource across source/destination. Explicit sinks are adult eating,
brood feeding, comb building and evaporation.

One feeding unit consumes one honey/nectar plus 0.2 pollen, reduces larval hunger
by 0.2, and restores 0.2 health points. Honey/nectar have equal substitution weight
for feeding, whereas nectar yields 0.7 honey-equivalent for adult energy and
reserve scoring. This is an explicit simplification, not chemical equivalence.

One repair unit consumes 0.3 honey and restores 0.05 integrity. Heating adds
0.6 °C per unit. Fanning mixes at most 25% toward ambient. Evaporation consumes
one water, cools by 0.8 °C and adds two humidity percentage points per unit.
An egg costs 0.5 honey and creates future demand, without directly improving
a vector component.

Guarding models entrance upkeep, not intruders or fighting. Grooming modifies
a local contamination proxy, not individual mites. Communication is an abstract
recruitment signal rather than a decoded waggle dance.

## 6. Background physics and time

At each round start the engine shuffles living adults with a seeded RNG; each
gets at most one decision. With N adults, global time advances by
round_minutes/N before each slot. A full round therefore advances by **one
round**, not N rounds. Bees dying before their slot are skipped without a call
or vector row. Sequential order effects remain despite shuffling.

Default 50 adults × six hours × four rounds/hour gives **1,200 decisions** if
all survive. max_actions bounds returned decisions even when duration is longer.
Population/duration generally scale API calls linearly; transport retries can
add requests without additional executed actions.

Actions approximate work over a round, while background evolution uses the
slot's global elapsed time. Effects are applied instantaneously at the slot
timestamp and become visible to subsequent bees. This is a serialized
approximation, not continuous-time interaction. Waiting for the API advances
wall time but never biological time.

World.advance updates:

- A sinusoidal 06:00–18:00 daylight cycle; ambient 20–32 °C and 75–55% humidity.
- Hive temperature via exponential mixing with ambient and neighboring
  compartments. Diffusion uses an old temperature map to avoid iteration-order
  heat creation. Humidity mixes toward ambient.
- Debris accumulation (0.08 units/hour/compartment), debris-dependent
  contamination growth and comb decay (0.0005/hour).
- Adults: six energy points lost/hour; health damage from thermal deviation
  beyond five degrees, contamination and exhausted energy; zero health is death.
- Larvae: 0.06 hunger/hour; damage from temperature deviation beyond three
  degrees, contamination and hunger at/above 0.95.
- Slowly replenishing daylight flowers with caps, finite pond water, and a small
  orchard toxin penalty on collection. Toxin is not carried through the food
  chain.

Passive rates, physiology, reserve anchors and thresholds are exposed in Config.
Other action coefficients are stated above and in World.execute. There is no
autonomous rescue policy: starvation, inaction, depletion and poor cooperation
can degrade the colony.

Food transfer conserves resource units. The thermal equations are phenomenological
mixing with ambient exchange and action inputs, not a heat-capacity model or
thermodynamic conservation solver.

## 7. Hidden vector and calibration

The stable component order is:

~~~text
[adult_survival, adult_vitality, adult_energy_sufficiency,
 carbohydrate_reserve, pollen_reserve, water_reserve,
 initial_brood_survival, brood_nutrition, thermal_suitability,
 humidity_suitability, hygiene, structural_integrity,
 pathogen_suppression, queen_vitality]
~~~

Define C(z)=min(1,max(0,z)). Non-finite values raise an error; final values are
clipped to [0,1] and rounded to ten decimals. **Zero is poorest and one best on
every declared engineering scale.** Anchors are fixed before the run, not
computed from observed minima/maxima.

N0 is starting adult count, B0 the initial larval cohort, H the 19 hive
compartments and HB the seven designated brood compartments. Dead cohort members
remain in denominators with zero contribution.

| Index / attribute | Formula | Endpoints |
| --- | --- | --- |
| 0 adult_survival | Living initial adults / N0 | 0 all dead; 1 all alive |
| 1 adult_vitality | Sum living initial-adult health / (100 N0) | 0 no surviving health; 1 all fully healthy |
| 2 adult_energy_sufficiency | Sum living initial-adult energy / (100 N0) | 0 none; 1 all adults full |
| 3 carbohydrate_reserve | C((in-hive honey + 0.7 × nectar) / sugar_target) | 0 none; 1 at least target |
| 4 pollen_reserve | C(in-hive pollen / pollen_target) | 0 none; 1 at least target |
| 5 water_reserve | C(in-hive water / water_target) | 0 none; 1 at least target |
| 6 initial_brood_survival | Living members of B0 / size(B0) | 0 initial cohort dead; 1 intact |
| 7 brood_nutrition | Mean over B0 of (1 − hunger), dead zero | 0 all starving/dead; 1 alive and satiated |
| 8 thermal_suitability | Mean over HB of C(1 − abs(T − 35)/10) | 0 all at least 10 °C from optimum; 1 all optimal |
| 9 humidity_suitability | Mean over HB of C(1 − abs(RH − 60)/40) | 0 all at least 40 points from optimum; 1 all optimal |
| 10 hygiene | Mean over H of C(1 − debris/10) | 0 all at/beyond poor anchor; 1 no debris |
| 11 structural_integrity | Mean comb-integrity fraction over H | 0 destroyed; 1 intact |
| 12 pathogen_suppression | Mean over H of C(1 − contamination/10) | 0 all at/beyond poor anchor; 1 none |
| 13 queen_vitality | Queen health / 100; zero if dead | 0 dead; 1 fully healthy |

In-hive resources include compartment stores and crops of **living adults
currently inside**. Cargo outside is uncredited until returned. Dead-bee cargo
is excluded. Moving outside can lower accessibility without destroying mass.
Queen health is counted both in the adult mean and in its separate attribute.

With the default 24-hour reserve horizon:

~~~text
sugar_target  = N0 × 0.30 × 24
pollen_target = N0 × 0.06 × 24
water_target  = N0 × 0.04 × 24
~~~

For 50 adults the targets are 360 sugar-equivalent, 72 pollen and 48 water units.
These are reference-demand assumptions, not guarantees of 24-hour biological
survival. Adults do not explicitly eat pollen or drink water here: pollen feeds
brood and water supports cooling. Brood/activity affect actual depletion.

Targets, optimum readings and poor-condition anchors are configurable. This is
engineering calibration, **not statistical calibration against empirical hives**.
Saturation masks extra reserves above target; raw states remain in metadata.

Fixed initial cohorts prevent births or death of hungry larvae from artificially
raising averages. New eggs consume resources and later need care but are not
part of cohort indices; reproductive output must also be studied in raw logs.
Microclimate weights designated brood compartments equally, including empty ones.
It measures maintained brood area rather than brood-count-weighted conditions.

## 8. Snapshot order, exports and failures

For every scheduled living bee:

1. Advance background time/physics by a global slot.
2. Construct its local observation and researcher-only pre-action vector.
3. Call its policy; live mode calls gpt-6-luna.
4. Parse, validate and apply or reject its action proposal.
5. **Immediately take one post-action vector.**
6. Append vector/audit, then update private memory and previous outcome.

Nothing advances the environment between action and snapshot. A parseable
schema-invalid or impossible proposal is an attempted action: it gets a vector
row and rejection outcome. A refusal, incomplete response, transport failure or
non-JSON response has no executable action: stop and preserve prior rows without
inventing a decision.

Simulation.run returns the complete ordered transcript:

~~~json
[
  {
    "timestamp": "2026-01-01T06:00:18.000000+00:00",
    "vector": [1.0, 0.9499, 0.7997, 0.5139, 0.5556, 0.5208, 1.0,
               0.7497, 0.8996, 0.9999, 0.85, 0.8, 0.9, 0.95]
  }
]
~~~

This example is illustrative, not measured. Timestamps are simulated ISO 8601
times with timezone. Vector order is VECTOR_NAMES and is saved in metadata.
The initial baseline is separate metadata, not an extra action row.

Every run creates a fresh directory and refuses to overwrite an existing one:

- vectors.jsonl: timestamp/vector object per line, flushed after each action;
- vectors.csv: timestamp plus 14 named columns, exported on normal completion
  or caught interruption;
- actions.jsonl: researcher-only action, outcome, bee, returned model/response
  identifier, usage, wall timestamp, pre-vector and deltas;
- metadata.json: configuration, calibration, initial/final raw state, baseline,
  prompt, schema, engine hash, reference cohorts, event count and stop status.

JSONL files are flushed but do not form an atomic database transaction: a process
kill/disk error between writes can leave an unmatched final row. Recover only
the common valid prefix. There is no per-row fsync. In-memory logs grow linearly;
this notebook targets moderate runs. Partial output is preserved but physical
checkpoint resume is not implemented.

Completed=true means the configured stopping condition was reached, which might
be max_actions or no living adults rather than the full duration. Inspect
stop_reason and final time. Metadata's final raw state may include background
time after the last recorded action if remaining scheduled bees die.

## 9. Collective improvement analysis

The notebook plots all attributes on the same scale. Simulation.comparison
reports final-minus-initial changes, improved/worsened dimensions and:

- Pareto improvement: positive changes, no negative changes;
- Pareto deterioration: negative changes, no positive changes;
- mixed tradeoffs: both positive and negative;
- unchanged: changes within numerical tolerance.

There is no privileged scalar hive welfare. Optional weighted_research_score
requires explicit non-negative researcher weights and never affects agent
prompts, rewards or decisions.

Observer-only audit decomposes change:

~~~text
background_delta(i) = pre_action(i) − post_action(i−1)
immediate_action_delta(i) = post_action(i) − pre_action(i)
~~~

Event one's previous vector is the baseline. Their sums reconstruct
final-minus-initial within rounding. Immediate effects differ from eventual
benefit: an action can incur future food costs or brood obligations. Background
evolution also depends on earlier actions. These observational logs alone do
not establish causal attribution.

Compare several seeds and matched initial conditions, an explicit baseline
controller, equal budgets and frozen calibration. Report per-attribute outcomes,
interaction/rejection rates and chosen weighting/cohort conventions. Changing
normalization targets changes the scale, not the agents' underlying performance.

Seeds reproduce the world and offline controller. Live responses remain
stochastic and aliases may evolve. Audits retain model/response identifiers and
usage; exact live replay would require applying recorded actions rather than
requesting new decisions.

## 10. Verification and scientific limits

Tests cover adjustable populations and round duration, one vector per proposal,
timestamp order, all normalized bounds/endpoints, distant-state privacy,
detached observations, personal memory, local/expiring messages, material
conservation, movement/caste/role constraints, brood feeding sinks, beneficial
and harmful heating, delta accounting, offline reproducibility, consistent
exports, partial-log preservation, overwrite protection and the OpenAI request
contract with a fake client.

The notebook is validated and run end-to-end offline. A fake client verifies
request arguments, not server behaviour. A live test requires the SDK, a key
and model access; without these, no live inference result is claimed.

Main limits: small population, coarse space/time, language-mediated cognition,
abstract communication, simplified nutrition, health and contamination,
incomplete worker lifecycle, absent drones/mating and no fitted demographic,
disease, pesticide or thermodynamic parameters. Multi-day runs are technically
configurable but not validated long-horizon colony simulations.

## References

1. Schmickl, T. & Crailsheim, K. (2008). *TaskSelSim: a model of the
   self-organization of the division of labour in honeybees*.
   [DOI:10.1080/13873950701846662](https://doi.org/10.1080/13873950701846662).
2. Duan, X., Wallis, D., Hatjina, F., Simon-Delso, N., Bruun Jensen, A. &
   Topping, C. J. (2022). *ApisRAM Formal Model Description*, EN-7184.
   [DOI:10.2903/sp.efsa.2022.EN-7184](https://doi.org/10.2903/sp.efsa.2022.EN-7184).
   [Full report](https://www.pollinera-horizon.eu/storage/app/uploads/public/668/52d/69c/66852d69c0ea0784169394.pdf).
3. Alves, T. (2018–2021). *Visualisation and Interactions with a Virtual Honey
   Bee Colony: Simulation, Complexity and Pedagogy* (SIMBACA).
   [Institutional project description](https://labsticc.fr/en/teams/inuit/past-phd-thesis/phd-thomas-alves).
4. Betti, M., LeClair, J., Wahl, L. M. & Zamir, M. (2017). *Bee++: An
   Object-Oriented, Agent-Based Simulator for Honey Bee Colonies*. Insects 8(1),31.
   [DOI:10.3390/insects8010031](https://doi.org/10.3390/insects8010031).
5. OpenAI. [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
   and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
   Checked 2026-10-31.
